fn init() {
    let state = #{};
    state.speed = 0.0;
    return state;
}

fn update(state, transform, dt) {
    let n4_0 = transform;
    let n5_0 = n4_0.translation;
    let n5_1 = n4_0.rotation;
    let n5_2 = n4_0.scale;
    let n6_0 = n5_1.x;
    let n6_1 = n5_1.y;
    let n6_2 = n5_1.z;
    let n1_0 = dt;
    let n2_0 = state.speed;
    let n3_0 = n1_0 * n2_0;
    let n7_0 = n6_1 + n3_0;
    let n8_0 = vec3(n6_0, n7_0, n6_2);
    let mut n9_0 = transform;
    n9_0.translation = n5_0;
    n9_0.rotation = n8_0;
    n9_0.scale = n5_2;
    transform = n9_0;
}
