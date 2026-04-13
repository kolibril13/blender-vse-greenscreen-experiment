import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from pathlib import Path
import os
import threading
import time
import traceback

import numpy as np

from . import corridorkey_loader
from .corridorkey_loader import debug_print

TREE_NAME = "CorridorKey GreenScreen"
TREE_IDNAME = "CorridorKeyGreenScreenTreeType"
SOCKET_IDNAME = "CorridorKeyColorSocket"


class CorridorKeyAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__ if __package__ else __name__.split(".")[0]

    corridorkey_engine_root: StringProperty(
        name="CorridorKey engine root",
        description=(
            "Path to the CorridorKey-Engine checkout (folder that contains CorridorKeyModule). "
            "Leave empty to use CORRIDORKEY_ENGINE_ROOT or the folder next to this add-on's parent: "
            "../CorridorKey-Engine. Place CorridorKey.pth in CorridorKeyModule/checkpoints/ there."
        ),
        default="",
        subtype="DIR_PATH",
        maxlen=1024,
    )

    corridorkey_backend: EnumProperty(
        name="Backend",
        description=(
            "Inference backend. Default is Torch so Apple Silicon uses PyTorch+MPS. "
            "Auto can pick MLX (separate runtime) which may ignore Device and behave differently in Blender."
        ),
        items=[
            ("AUTO", "Auto", "Detect MLX on Apple Silicon or CUDA optimized path"),
            ("torch", "Torch", "Standard PyTorch engine (recommended in Blender)"),
            ("torch_optimized", "Torch optimized", "Optimized engine when CUDA is available"),
            ("mlx", "MLX", "Apple Silicon MLX (requires corridorkey_mlx and .safetensors)"),
        ],
        default="torch",
    )

    corridorkey_device: StringProperty(
        name="Device",
        description=(
            'Torch device: "mps" (Apple GPU), "cuda", or "cpu". '
            "Leave empty for automatic CUDA → MPS → CPU. "
            "Note: CorridorKey used to default to CPU when empty and could take 10+ minutes per frame."
        ),
        default="",
        maxlen=64,
    )

    corridorkey_img_size: IntProperty(
        name="Model resolution",
        description=(
            "Internal square size passed to create_engine (img_size). "
            "Lower = faster (try 1024 in Blender); 2048 is slower."
        ),
        default=1024,
        min=256,
        max=8192,
    )

    def draw(self, _context):
        layout = self.layout
        layout.prop(self, "corridorkey_engine_root")
        layout.prop(self, "corridorkey_backend")
        layout.prop(self, "corridorkey_device")
        layout.prop(self, "corridorkey_img_size")


class CorridorKeyGreenScreenTree(bpy.types.NodeTree):
    bl_idname = TREE_IDNAME
    bl_label = TREE_NAME
    bl_icon = "NODETREE"


class CorridorKeyColorSocket(bpy.types.NodeSocket):
    bl_idname = SOCKET_IDNAME
    bl_label = "CorridorKey Color"

    default_value: FloatVectorProperty(
        name="Color",
        subtype="COLOR",
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0),
    )

    def draw(self, _context, layout, _node, text):
        if self.is_output or self.is_linked:
            layout.label(text=text)
        else:
            layout.prop(self, "default_value", text=text)

    def draw_color(self, _context, _node):
        return (0.15, 0.75, 0.2, 1.0)


class CorridorKeyBaseNode:
    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == TREE_IDNAME


class CorridorKeyInputNode(CorridorKeyBaseNode, bpy.types.Node):
    bl_idname = "CorridorKeyInputNode"
    bl_label = "Input Image"

    use_vse: BoolProperty(
        name="Use VSE",
        description="Read image from the currently visible VSE strip at current frame",
        default=True,
    )
    source_image: PointerProperty(type=bpy.types.Image, name="Fallback Image")

    def init(self, _context):
        output = self.outputs.new(SOCKET_IDNAME, "Image")
        output.default_value = (0.0, 1.0, 0.0, 1.0)

    def draw_buttons(self, _context, layout):
        layout.prop(self, "use_vse")
        layout.prop(self, "source_image")

    def resolve_image(self, scene):
        if self.use_vse:
            vse_image = _load_current_vse_image(scene)
            if vse_image is not None:
                return vse_image
        return self.source_image


class CorridorKeyProcessNode(CorridorKeyBaseNode, bpy.types.Node):
    bl_idname = "CorridorKeyProcessNode"
    bl_label = "CorridorKey Process"

    despill_strength: FloatProperty(
        name="Despill",
        description="Green spill removal strength (0–1)",
        default=0.5,
        min=0.0,
        max=1.0,
    )
    refiner_scale: FloatProperty(
        name="Refiner scale",
        description="Multiplier for refiner output deltas",
        default=1.0,
        min=0.0,
        max=4.0,
    )
    auto_despeckle: BoolProperty(
        name="Auto despeckle",
        description="Remove small disconnected alpha islands",
        default=True,
    )
    despeckle_size: IntProperty(
        name="Despeckle size",
        description="Minimum connected area (pixels) to keep when despeckling",
        default=400,
        min=1,
        max=10000,
    )
    use_image_alpha_hint: BoolProperty(
        name="Use image alpha as hint",
        description="Use the source alpha channel as CorridorKey alpha hint when it varies; "
        "otherwise build a rough green-screen hint from RGB",
        default=True,
    )
    force_linear_input: BoolProperty(
        name="Force linear input",
        description="If enabled, treat input as linear RGB before the engine (matches float/HDR). "
        "If off, float images are still treated as linear automatically.",
        default=False,
    )

    def init(self, _context):
        self.inputs.new(SOCKET_IDNAME, "Image")
        self.outputs.new(SOCKET_IDNAME, "Image")

    def draw_buttons(self, _context, layout):
        layout.prop(self, "despill_strength")
        layout.prop(self, "refiner_scale")
        layout.prop(self, "auto_despeckle")
        layout.prop(self, "despeckle_size")
        layout.prop(self, "use_image_alpha_hint")
        layout.prop(self, "force_linear_input")


class CorridorKeyOutputNode(CorridorKeyBaseNode, bpy.types.Node):
    bl_idname = "CorridorKeyOutputNode"
    bl_label = "Output Image"

    output_image: PointerProperty(type=bpy.types.Image, name="Image")

    def init(self, _context):
        self.inputs.new(SOCKET_IDNAME, "Image")

    def draw_buttons(self, _context, layout):
        layout.prop(self, "output_image")


def _strip_covers_frame(strip, frame):
    return strip.frame_final_start <= frame < strip.frame_final_end


def _active_vse_strips(scene):
    editor = scene.sequence_editor
    if editor is None:
        return []

    strip_collections = (
        getattr(editor, "strips_all", None),
        getattr(editor, "strips", None),
        getattr(editor, "sequences_all", None),
        getattr(editor, "sequences", None),
    )

    for strips in strip_collections:
        if strips is None:
            continue
        return [s for s in strips if not getattr(s, "mute", False)]

    return []


def _pick_best_vse_strip(scene):
    frame = scene.frame_current
    candidates = [s for s in _active_vse_strips(scene) if _strip_covers_frame(s, frame)]
    if not candidates:
        return None
    candidates.sort(key=lambda s: (s.channel, s.frame_final_start), reverse=True)
    return candidates[0]


def _ensure_vse_editor(scene):
    editor = scene.sequence_editor
    if editor is None:
        editor = scene.sequence_editor_create()
    return editor


def _strip_collection(editor):
    for attr in ("strips", "sequences"):
        collection = getattr(editor, attr, None)
        if collection is not None:
            return collection
    return None


def _remove_strip(editor, strip):
    for attr in ("strips", "sequences"):
        collection = getattr(editor, attr, None)
        if collection is not None and hasattr(collection, "remove"):
            collection.remove(strip)
            return


def _export_image_to_temp_png(image, scene):
    tmp_dir = bpy.app.tempdir or str(Path.home())
    filename = f"corridorkey_{scene.frame_current:06d}.png"
    filepath = str(Path(tmp_dir) / filename)

    original_filepath_raw = image.filepath_raw
    original_format = image.file_format
    try:
        image.filepath_raw = filepath
        image.file_format = "PNG"
        image.save()
    finally:
        image.filepath_raw = original_filepath_raw
        image.file_format = original_format

    return filepath


def _add_image_strip(editor, name, filepath, channel, frame_start):
    collection = _strip_collection(editor)
    if collection is None:
        raise RuntimeError("Unable to access Sequencer strip collection")

    if hasattr(collection, "new_image"):
        return collection.new_image(
            name=name,
            filepath=filepath,
            channel=channel,
            frame_start=frame_start,
        )

    if hasattr(collection, "new"):
        return collection.new(
            name=name,
            type="IMAGE",
            filepath=filepath,
            channel=channel,
            frame_start=frame_start,
        )

    raise RuntimeError("Sequencer API has no supported method to create image strips")


def place_output_strip_above(scene, output_image):
    source_strip = _pick_best_vse_strip(scene)
    if source_strip is None:
        raise RuntimeError("No visible VSE strip at current frame")

    editor = _ensure_vse_editor(scene)
    strip_name = "CorridorKey Output"
    export_path = _export_image_to_temp_png(output_image, scene)
    target_channel = int(source_strip.channel) + 1
    start_frame = int(scene.frame_current)

    for strip in list(_active_vse_strips(scene)):
        if strip.name == strip_name:
            _remove_strip(editor, strip)

    created = _add_image_strip(
        editor=editor,
        name=strip_name,
        filepath=export_path,
        channel=target_channel,
        frame_start=start_frame,
    )
    return created


def _image_path_from_strip(strip, frame):
    if strip.type != "IMAGE":
        return None
    if not strip.elements:
        return None

    frame_offset = int(frame - strip.frame_start)
    index = max(0, min(frame_offset, len(strip.elements) - 1))
    filename = strip.elements[index].filename
    directory = bpy.path.abspath(strip.directory)
    return bpy.path.abspath(f"{directory}/{filename}")


def _load_current_vse_image(scene):
    strip = _pick_best_vse_strip(scene)
    if strip is None:
        return None

    if strip.type == "IMAGE":
        image_path = _image_path_from_strip(strip, scene.frame_current)
        if image_path is None:
            return None
        try:
            return bpy.data.images.load(image_path, check_existing=True)
        except RuntimeError:
            return None

    if strip.type == "SCENE":
        scene_image_name = "Render Result"
        return bpy.data.images.get(scene_image_name)

    return None


def _image_size(image):
    width = max(1, int(image.size[0]))
    height = max(1, int(image.size[1]))
    return width, height


def _ensure_output_image(name, width, height, float_buffer=False):
    image = bpy.data.images.get(name)
    if image is None:
        image = bpy.data.images.new(name=name, width=width, height=height, alpha=True, float_buffer=float_buffer)
    elif int(image.size[0]) != width or int(image.size[1]) != height:
        bpy.data.images.remove(image)
        image = bpy.data.images.new(name=name, width=width, height=height, alpha=True, float_buffer=float_buffer)
    return image


def _blender_pixels_to_numpy_rgba(image):
    """Return RGBA float32 [H,W,4], rows top-first (numpy convention)."""
    width, height = _image_size(image)
    flat = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(flat)
    rgba = flat.reshape((height, width, 4))
    return rgba[::-1].copy()


def _rough_green_screen_mask(rgb: np.ndarray) -> np.ndarray:
    """Build a coarse alpha hint when no usable alpha is present (numpy only)."""
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    max_rb = np.maximum(r, b)
    green_excess = g - max_rb
    mask = 1.0 - np.clip((green_excess + 0.05) / 0.35, 0.0, 1.0)
    return mask.astype(np.float32)


def _alpha_hint_from_image(rgb: np.ndarray, alpha: np.ndarray, use_alpha: bool) -> np.ndarray:
    if use_alpha and alpha.size > 0:
        spread = float(np.max(alpha) - np.min(alpha))
        if spread > 0.02:
            return np.clip(alpha, 0.0, 1.0).astype(np.float32)
    return _rough_green_screen_mask(rgb)


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear-light RGB to sRGB (standard gamma ~2.2 curve)."""
    linear = np.clip(linear, 0.0, 1.0)
    srgb = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    return srgb.astype(np.float32)


def _input_is_linear(image, process_node: CorridorKeyProcessNode) -> bool:
    if process_node.force_linear_input:
        return True
    return bool(getattr(image, "is_float", False))


def _run_corridor_key(source_image, process_node: CorridorKeyProcessNode):
    debug_print("pixels: reading Blender image into numpy…")
    t0 = time.perf_counter()
    rgba = _blender_pixels_to_numpy_rgba(source_image)
    debug_print(f"pixels: numpy RGBA shape={rgba.shape} dtype={rgba.dtype} ({time.perf_counter() - t0:.3f}s)")
    rgb = np.ascontiguousarray(rgba[:, :, :3].astype(np.float32, copy=False))
    alpha = rgba[:, :, 3]

    mask_linear = _alpha_hint_from_image(
        rgb,
        alpha,
        process_node.use_image_alpha_hint,
    )
    mask_linear = np.ascontiguousarray(mask_linear)
    debug_print(f"alpha hint mask: mean={float(mask_linear.mean()):.4f}")

    debug_print("get_engine() …")
    t1 = time.perf_counter()
    engine = corridorkey_loader.get_engine()
    debug_print(f"get_engine() done ({time.perf_counter() - t1:.3f}s)")
    input_linear = _input_is_linear(source_image, process_node)
    debug_print(f"process_frame: input_is_linear={input_linear}")

    interval = float(os.environ.get("CORRIDORKEY_HEARTBEAT_SEC", "10"))
    stop_hb = threading.Event()

    def heartbeat() -> None:
        step = 0
        while not stop_hb.wait(interval):
            step += 1
            debug_print(
                f"process_frame: still running (~{step * interval:.0f}s) — "
                "not frozen if this repeats; on CPU use Device=mps or lower Model resolution"
            )

    hb = threading.Thread(target=heartbeat, daemon=True)
    debug_print(
        "process_frame: running inference (GPU is usually <2 min; CPU at 2048 can exceed 15 min)…"
    )
    t2 = time.perf_counter()
    hb.start()
    try:
        result = engine.process_frame(
            rgb,
            mask_linear,
            refiner_scale=process_node.refiner_scale,
            input_is_linear=input_linear,
            despill_strength=process_node.despill_strength,
            auto_despeckle=process_node.auto_despeckle,
            despeckle_size=process_node.despeckle_size,
        )
    finally:
        stop_hb.set()
    debug_print(f"process_frame: returned ({time.perf_counter() - t2:.3f}s)")
    if isinstance(result, list):
        result = result[0]

    debug_print(f"result keys: {sorted(result.keys())}")

    # `processed` is linear premultiplied RGBA — looks dark in Blender.
    # Build sRGB straight RGBA from `processed` by un-premultiplying and
    # converting linear → sRGB so the VSE shows correct colors + transparency.
    processed = result["processed"].astype(np.float32, copy=False)
    debug_print(f"raw processed shape={processed.shape} ndim={processed.ndim}")

    # GPU postprocess returns CHW [4, H, W] in BGRA order; CPU returns HWC [H, W, 4] in RGBA.
    if processed.ndim == 3 and processed.shape[0] in (1, 3, 4) and processed.shape[0] < processed.shape[1]:
        processed = np.transpose(processed, (1, 2, 0))
        processed = processed[:, :, [2, 1, 0, 3]]
        debug_print(f"transposed CHW→HWC + BGRA→RGBA: {processed.shape}")

    alpha = processed[:, :, 3:4]
    rgb_premul_lin = processed[:, :, :3]

    eps = 1e-7
    safe_alpha = np.maximum(alpha, eps)
    rgb_straight_lin = rgb_premul_lin / safe_alpha
    rgb_straight_lin = np.where(alpha > eps, rgb_straight_lin, 0.0)
    rgb_straight_lin = np.clip(rgb_straight_lin, 0.0, 1.0)

    rgb_srgb = _linear_to_srgb(rgb_straight_lin)

    out_rgba = np.concatenate([rgb_srgb, alpha], axis=-1).astype(np.float32)
    out_rgba = np.flipud(out_rgba)
    flat = out_rgba.reshape(-1)
    debug_print(
        f"output: sRGB straight RGBA shape={out_rgba.shape} "
        f"alpha range=[{float(alpha.min()):.3f}, {float(alpha.max()):.3f}]"
    )
    return flat


def _friendly_inference_error(exc: Exception) -> str:
    if isinstance(exc, ModuleNotFoundError):
        return (
            f"Missing Python module: {exc.name}. Install CorridorKey dependencies "
            "into Blender's Python (at least torch, torchvision, timm, numpy, opencv-python, huggingface_hub)."
        )
    if isinstance(exc, FileNotFoundError) and "CorridorKey-Engine" in str(exc):
        return str(exc)
    return (
        "CorridorKey inference failed: "
        f"{exc.__class__.__name__}: {exc}"
    )


def evaluate_tree(tree, scene):
    debug_print(
        f"evaluate_tree: frame={scene.frame_current} "
        f"(set CORRIDORKEY_VERBOSE=0 to silence; logs go to stderr when Blender runs from a terminal)"
    )
    input_node = next((n for n in tree.nodes if n.bl_idname == "CorridorKeyInputNode"), None)
    process_node = next((n for n in tree.nodes if n.bl_idname == "CorridorKeyProcessNode"), None)
    output_node = next((n for n in tree.nodes if n.bl_idname == "CorridorKeyOutputNode"), None)
    if input_node is None or process_node is None or output_node is None:
        raise RuntimeError("Tree requires Input Image, CorridorKey Process, and Output Image nodes")

    source_image = input_node.resolve_image(scene)
    if source_image is None:
        raise RuntimeError("No VSE image at current frame and no fallback image selected")

    width, height = _image_size(source_image)
    use_float = bool(getattr(source_image, "is_float", False))
    debug_print(
        f"source image: name={source_image.name!r} size={width}x{height} float={use_float}"
    )
    output_image = _ensure_output_image(
        name=f"{TREE_NAME} Output",
        width=width,
        height=height,
        float_buffer=use_float,
    )

    try:
        flat = _run_corridor_key(source_image, process_node)
    except Exception as exc:
        debug_print(f"ERROR: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise RuntimeError(_friendly_inference_error(exc)) from exc

    debug_print("assigning output_image.pixels …")
    output_image.pixels.foreach_set(flat)
    output_image.update()
    debug_print("evaluate_tree: done")

    if len(flat) >= 4:
        process_node.outputs["Image"].default_value = (
            float(flat[0]),
            float(flat[1]),
            float(flat[2]),
            float(flat[3]),
        )
    output_node.output_image = output_image
    return output_image


class CORRIDORKEY_OT_new_tree(bpy.types.Operator):
    bl_idname = "node.corridor_key_new_tree"
    bl_label = "New CorridorKey GreenScreen Tree"
    bl_description = "Create and open a new CorridorKey GreenScreen node tree"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.space_data and context.space_data.type == "NODE_EDITOR"

    def execute(self, context):
        tree = bpy.data.node_groups.new(TREE_NAME, TREE_IDNAME)

        input_node = tree.nodes.new("CorridorKeyInputNode")
        input_node.location = (-500, 0)

        process_node = tree.nodes.new("CorridorKeyProcessNode")
        process_node.location = (-120, 0)

        output_node = tree.nodes.new("CorridorKeyOutputNode")
        output_node.location = (260, 0)

        tree.links.new(input_node.outputs["Image"], process_node.inputs["Image"])
        tree.links.new(process_node.outputs["Image"], output_node.inputs["Image"])

        space = context.space_data
        try:
            space.tree_type = TREE_IDNAME
        except AttributeError:
            pass
        space.node_tree = tree
        return {"FINISHED"}


class CORRIDORKEY_OT_add_node(bpy.types.Operator):
    bl_idname = "node.corridor_key_add_node"
    bl_label = "Add CorridorKey Node"
    bl_options = {"REGISTER", "UNDO"}

    node_type: StringProperty(name="Node Type")

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return (
            space
            and space.type == "NODE_EDITOR"
            and space.edit_tree is not None
            and space.edit_tree.bl_idname == TREE_IDNAME
        )

    def execute(self, context):
        tree = context.space_data.edit_tree
        node = tree.nodes.new(self.node_type)
        node.location = context.space_data.cursor_location
        return {"FINISHED"}


class CORRIDORKEY_OT_evaluate_tree(bpy.types.Operator):
    bl_idname = "node.corridor_key_evaluate_tree"
    bl_label = "Evaluate CorridorKey Tree"
    bl_description = "Read current VSE frame and run CorridorKey process_frame on the image"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return (
            space
            and space.type == "NODE_EDITOR"
            and space.edit_tree is not None
            and space.edit_tree.bl_idname == TREE_IDNAME
        )

    def execute(self, context):
        debug_print("operator: Evaluate CorridorKey Tree — start")
        try:
            output_image = evaluate_tree(context.space_data.edit_tree, context.scene)
            debug_print("operator: placing VSE strip …")
            output_strip = place_output_strip_above(context.scene, output_image)
        except RuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        self.report(
            {"INFO"},
            f"Evaluated to image: {output_image.name}; placed strip '{output_strip.name}' on channel {output_strip.channel}",
        )
        debug_print("operator: Evaluate CorridorKey Tree — finished OK")
        return {"FINISHED"}


class CORRIDORKEY_PT_tools(bpy.types.Panel):
    bl_label = "CorridorKey"
    bl_space_type = "NODE_EDITOR"
    bl_region_type = "UI"
    bl_category = "CorridorKey"

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return (
            space
            and space.type == "NODE_EDITOR"
            and space.edit_tree is not None
            and space.edit_tree.bl_idname == TREE_IDNAME
        )

    def draw(self, context):
        layout = self.layout
        layout.operator(CORRIDORKEY_OT_evaluate_tree.bl_idname, icon="SEQ_SEQUENCER")
        layout.label(text=f"Frame: {context.scene.frame_current}")


def draw_add_menu(self, context):
    layout = self.layout
    layout.separator()
    layout.operator(
        CORRIDORKEY_OT_new_tree.bl_idname,
        text="CorridorKey GreenScreen Tree",
        icon="NODETREE",
    )

    space = context.space_data
    if not (space and space.type == "NODE_EDITOR" and space.edit_tree):
        return
    if space.edit_tree.bl_idname != TREE_IDNAME:
        return

    layout.separator()

    op = layout.operator(CORRIDORKEY_OT_add_node.bl_idname, text="Input Image", icon="IMAGE_DATA")
    op.node_type = "CorridorKeyInputNode"
    op = layout.operator(
        CORRIDORKEY_OT_add_node.bl_idname,
        text="CorridorKey Process",
        icon="IMAGE_DATA",
    )
    op.node_type = "CorridorKeyProcessNode"
    op = layout.operator(CORRIDORKEY_OT_add_node.bl_idname, text="Output Image", icon="IMAGE_DATA")
    op.node_type = "CorridorKeyOutputNode"


def register():
    corridorkey_loader.clear_engine_cache()
    bpy.types.NODE_MT_add.append(draw_add_menu)


def unregister():
    bpy.types.NODE_MT_add.remove(draw_add_menu)
    corridorkey_loader.clear_engine_cache()
