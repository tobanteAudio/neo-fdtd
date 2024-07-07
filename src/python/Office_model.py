from common.room_builder import RoomBuilder, find_third_vertex

L = 6.0
W = 3.65
H = 3.12

src_height = 1.2
src_backwall = 1
src_distance = 1.75
src_left = [W/2-src_distance/2, L-src_backwall, src_height]
src_right = [W/2+src_distance/2, L-src_backwall, src_height]

l1, l2 = find_third_vertex(src_left, src_right)
listener = l1 if l1[1] < l2[1] else l2

producer_sit = listener.copy()
producer_sit[1] = 1
producer_sit[2] = 0.9

producer_stand = listener.copy()
producer_stand[1] = 1

builder = RoomBuilder(W, L, H, wall_color=[255, 255, 255])
builder.with_colors({
    "Ceiling": [200, 200, 200],
    "Floor": [151, 134, 122],
    "Panel": [111, 55, 10],
    "Table": [130, 75, 25],
    "Sofa": [25, 25, 25],
    "Walls": [255, 255, 255],
})

builder.add_box("Panel", [2.5, 0.1, 1.5], [W/2-2.5/2, L-0.3, 0.75])
builder.add_box("Panel", [0.1, 2.5, 1.5], [0.2, L-2.5-0.5, 0.75])
builder.add_box("Panel", [0.1, 2.5, 1.5], [W-0.1-0.2, L-2.5-0.5, 0.75])
builder.add_box("Panel", [2.5, 2.0, 0.1], [W/2-2.5/2, L-2-0.5, H-0.1-0.3])
builder.add_box("Panel", [2.5, 2.0, 0.1], [W/2-2.5/2, L-2-2.1-0.5, H-0.1-0.3])
builder.add_box("Panel", [2.5, 0.1, 1.5], [W/2-2.5/2, 0.3, 0.75])

builder.add_box("Sofa", [2.52, 0.98, 0.48], [W/2-2.52/2, 0.1, 0.05])
builder.add_box("Table", [1.8, 0.8, 0.02], [W/2-1.8/2, listener[1]+0.4, 0.7])

builder.add_source("Speaker Left", src_left)
builder.add_source("Speaker Right", src_right)
builder.add_receiver("Engineer", listener.tolist())

builder.add_receiver("Producer Sitting", producer_sit.tolist())
builder.add_receiver("Producer Standing", producer_stand.tolist())

model_file = '../../data/models/Office/model.json'
builder.build(model_file)
