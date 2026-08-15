import struct

with open("assets/test_triangle.mesh", "rb") as f:
    data = f.read()

# Header
fmt = "<4sIIIII3f3fI"
magic, version, flags, v_count, i_count, m_count, min_x, min_y, min_z, max_x, max_y, max_z, pad = struct.unpack_from(fmt, data, 0)

print(f"Header: v={v_count}, i={i_count}, m={m_count}")

# Vertices
v_offset = 52
print(f"Vertices (offset={v_offset}):")
for i in range(v_count):
    v_data = data[v_offset + i * 64 : v_offset + (i+1) * 64]
    px, py, pz, nx, ny, nz, u, v = struct.unpack_from("<3f3f2f", v_data, 0)
    print(f"  v{i}: pos=[{px:.2f}, {py:.2f}, {pz:.2f}] normal=[{nx:.2f}, {ny:.2f}, {nz:.2f}] uv=[{u:.2f}, {v:.2f}]")

# Indices
i_offset = v_offset + v_count * 64
print(f"Indices (offset={i_offset}):")
for i in range(i_count):
    idx = struct.unpack_from("<I", data, i_offset + i * 4)[0]
    print(f"  i{i}: {idx}")

# Meshlets
m_offset = i_offset + i_count * 4
print(f"Meshlet (offset={m_offset}):")
for i in range(m_count):
    m_data = data[m_offset + i * 48 : m_offset + (i+1) * 48]
    cx, cy, cz, r, ax, ay, az, cutoff, i_off, t_count = struct.unpack_from("<3ff3ffII", m_data, 0)
    print(f"  m{i}: center=[{cx:.2f}, {cy:.2f}, {cz:.2f}], radius={r:.2f}, cone_axis=[{ax:.2f}, {ay:.2f}, {az:.2f}], cone_cutoff={cutoff:.2f}, idx={i_off}, tri={t_count}")
