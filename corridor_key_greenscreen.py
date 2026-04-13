import bpy
from bpy.props import BoolProperty, FloatProperty, FloatVectorProperty, PointerProperty, StringProperty
from pathlib import Path

TREE_NAME = "CorridorKey GreenScreen"
TREE_IDNAME = "CorridorKeyGreenScreenTreeType"
SOCKET_IDNAME = "CorridorKeyColorSocket"


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


class FakeCorridorKeyPlaceholderNode(CorridorKeyBaseNode, bpy.types.Node):
    bl_idname = "FakeCorridorKeyPlaceholderNode"
    bl_label = "FakeCorridorKeyPlaceholder"

    fake_threshold: FloatProperty(
        name="Fake Threshold",
        description="Placeholder control for future key threshold",
        default=0.5,
        min=0.0,
        max=1.0,
    )
    fake_softness: FloatProperty(
        name="Fake Softness",
        description="Placeholder control for future edge softness",
        default=0.25,
        min=0.0,
        max=1.0,
    )
    fake_despill: FloatProperty(
        name="Fake Despill",
        description="Placeholder control for future despill strength",
        default=0.3,
        min=0.0,
        max=1.0,
    )

    def init(self, _context):
        self.inputs.new(SOCKET_IDNAME, "Image")
        self.outputs.new(SOCKET_IDNAME, "Image")

    def draw_buttons(self, _context, layout):
        layout.prop(self, "fake_threshold")
        layout.prop(self, "fake_softness")
        layout.prop(self, "fake_despill")

    def _get_input_color(self):
        input_socket = self.inputs.get("Image")
        if input_socket is None:
            return (0.0, 0.0, 0.0, 1.0)

        if input_socket.is_linked and input_socket.links:
            from_socket = input_socket.links[0].from_socket
            if hasattr(from_socket, "default_value"):
                return tuple(from_socket.default_value)

        return tuple(input_socket.default_value)

    def update(self):
        rgba = list(self._get_input_color())
        rgba[1] = 0.0
        self.outputs["Image"].default_value = rgba


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

    # Blender 5.1 API uses strips/strips_all instead of sequences/sequences_all.
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

    # Replace prior auto-generated strip so reevaluating updates cleanly.
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


def _run_fake_corridor_key(source_image):
    pixels = list(source_image.pixels[:])
    for i in range(1, len(pixels), 4):
        pixels[i] = 0.0
    return pixels


def evaluate_tree(tree, scene):
    input_node = next((n for n in tree.nodes if n.bl_idname == "CorridorKeyInputNode"), None)
    fake_node = next((n for n in tree.nodes if n.bl_idname == "FakeCorridorKeyPlaceholderNode"), None)
    output_node = next((n for n in tree.nodes if n.bl_idname == "CorridorKeyOutputNode"), None)
    if input_node is None or fake_node is None or output_node is None:
        raise RuntimeError("Tree requires Input Image, FakeCorridorKeyPlaceholder, and Output Image nodes")

    source_image = input_node.resolve_image(scene)
    if source_image is None:
        raise RuntimeError("No VSE image at current frame and no fallback image selected")

    width, height = _image_size(source_image)
    output_image = _ensure_output_image(
        name=f"{TREE_NAME} Output",
        width=width,
        height=height,
        float_buffer=getattr(source_image, "is_float", False),
    )
    output_pixels = _run_fake_corridor_key(source_image)
    output_image.pixels[:] = output_pixels
    output_image.update()

    if output_pixels:
        fake_node.outputs["Image"].default_value = (
            output_pixels[0],
            output_pixels[1],
            output_pixels[2],
            output_pixels[3],
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

        fake_key = tree.nodes.new("FakeCorridorKeyPlaceholderNode")
        fake_key.location = (-120, 0)

        output_node = tree.nodes.new("CorridorKeyOutputNode")
        output_node.location = (260, 0)

        tree.links.new(input_node.outputs["Image"], fake_key.inputs["Image"])
        tree.links.new(fake_key.outputs["Image"], output_node.inputs["Image"])

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
    bl_description = "Read current VSE frame and evaluate the custom CorridorKey tree"
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
        try:
            output_image = evaluate_tree(context.space_data.edit_tree, context.scene)
            output_strip = place_output_strip_above(context.scene, output_image)
        except RuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        self.report(
            {"INFO"},
            f"Evaluated to image: {output_image.name}; placed strip '{output_strip.name}' on channel {output_strip.channel}",
        )
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
        text="FakeCorridorKeyPlaceholder",
        icon="NODE",
    )
    op.node_type = "FakeCorridorKeyPlaceholderNode"
    op = layout.operator(CORRIDORKEY_OT_add_node.bl_idname, text="Output Image", icon="IMAGE_DATA")
    op.node_type = "CorridorKeyOutputNode"


def register():
    bpy.types.NODE_MT_add.append(draw_add_menu)


def unregister():
    bpy.types.NODE_MT_add.remove(draw_add_menu)
