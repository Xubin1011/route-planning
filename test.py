import folium

def create_marker(map_object, latitude, longitude, stay):
    # 自定义HTML图标，显示字母"P"
    html_icon = folium.DivIcon(
        html=f'<div style="font-size: 16px; color: blue;">P</div>'
    )

    # 创建标记并添加到地图
    folium.Marker(
        location=[latitude, longitude],
        popup=f'Latitude: {latitude}<br>Longitude: {longitude}<br>Stay: {stay} mins',
        icon=html_icon
    ).add_to(map_object)

# 创建一个地图对象
map_object = folium.Map(location=[52.5200, 13.4050], zoom_start=12)

# 使用 create_marker 函数创建标记
create_marker(map_object, 52.5253, 13.3694, 30)

# 保存地图为 HTML 文件
map_object.save("map_with_marker.html")
