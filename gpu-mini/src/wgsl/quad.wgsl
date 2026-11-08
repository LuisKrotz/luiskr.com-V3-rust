// quad.wgsl — the shared quad-corner table for vertex-less geometry.
//
// Widgets with real quads (ring fills, instanced glyph cells) still
// carry no vertex BUFFER: the vertex shader indexes this 6-corner
// unit square by `vertex_index` (two triangles: 0,1,2 / 3,4,5) and
// scales/offsets the result in shader space — one draw(6, instances)
// call paints every copy.

/// quad_corner — unit-square corner for `vertex_index` 0..5, in uv
/// order bottom-left → bottom-right → top-left → top-left →
/// bottom-right → top-right (counter-clockwise pairs).
fn quad_corner(vertex_index: u32) -> vec2<f32> {
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
    );

    return corners[vertex_index];
}
