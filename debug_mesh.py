import struct

with open("assets/test_triangle.mesh", "rb") as f:
    data = f.read()

# MeshHeader: 52 bytes
# magic: 4, version: 4, flags: 4, v_count: 4, i_count: 4, m_count: 4
# aabb_min: 12, aabb_max: 12, pad: 4
fmt = "<4sIIIII3f3fI"
magic, version, flags, v_count, i_count, m_count, min_x, min_y, min_z, max_x, max_y, max_z, pad = struct.unpack_from(fmt, data, 0)

print(f"Header: v={v_count}, i={i_count}, m={m_count}")
print(f"AABB: [{min_x:.2f}, {min_y:.2f}, {min_z:.2f}] .. [{max_x:.2f}, {max_y:.2f}, {max_z:.2f}]")

# offset to meshlet
offset = 52 + v_count * 64 + i_count * 4
print(f"Meshlet offset: {offset}")

# MeshletData: 48 bytes
# center: 12, radius: 4, cone_axis: 12, cone_cutoff: 4, index_offset: 4, triangle_count: 4, pad: 8
m_fmt = "<3ff3ffII8s"
cx, cy, cz, r, ax, ay, az, cutoff, i_off, t_count, _ = struct.unpack_from(m_fmt, data, offset)
print(f"Meshlet: center=[{cx:.2f}, {cy:.2f}, {cz:.2f}], radius={r:.2f}, cone_axis=[{ax:.2f}, {ay:.2f}, {az:.2f}], cone_cutoff={cutoff:.2f}, index_offset={i_off}, triangle_count={t_count}")
