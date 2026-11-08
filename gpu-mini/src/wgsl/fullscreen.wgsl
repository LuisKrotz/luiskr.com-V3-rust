// fullscreen.wgsl — the shared vertex stage for SDF widgets.
//
// Three vertices at (-1,-1), (3,-1), (-1,3) form ONE triangle covering
// the entire clip plane: any pixel in [-1,1]² lies inside it (the two
// "3.0" corners just extend the triangle past the edges — everything
// beyond the viewport is clipped free by the rasterizer).
//
// Why not a 4-vertex quad? A quad is two triangles sharing a diagonal;
// some GPUs rasterize the shared edge twice → the fragment shader could
// run twice on the seam (visible banding in additive blends). One
// triangle = no seam, one fragment per pixel, guaranteed.
//
// The render-pass scissor rect (set CPU-side by gpu_mini::draw) clips
// this to the widget's rect, so only the widget's pixels are paid for.

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );

    return vec4<f32>(positions[vertex_index], 0.0, 1.0);
}
