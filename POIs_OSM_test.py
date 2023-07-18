import requests
import xml.etree.ElementTree as ET

#osm api, obtain information, output xml
def query_osm_api(top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon):
    url = "https://www.overpass-api.de/api/interpreter"
    query = f"""
    [out:xml];
    (
        node["amenity"="parking"]({bottom_right_lat},{top_left_lon},{top_left_lat},{bottom_right_lon});
        
       
    );
    out meta;
    """
    response = requests.post(url, data=query)
    return response.content

#extract required information from response
def extract_osm_xml(xml_data):
    root = ET.fromstring(xml_data)
    data = []
    for node in root.iter("node"):
        location_type = node.attrib.get("amenity", "")
        lat = node.attrib.get("lat", "")
        lon = node.attrib.get("lon", "")
        altitude = node.attrib.get("ele", "")
        charging_info = node.attrib.get("charging", "")
        data.append([location_type, lat, lon, altitude, charging_info])
    return data

#create xml to output
def create_xml_output(data):
    root = ET.Element("Locations")
    for item in data:
        location_type, lat, lon, altitude, charging_info = item
        location = ET.SubElement(root, "Location", type=location_type)
        ET.SubElement(location, "Latitude").text = str(lat)
        ET.SubElement(location, "Longitude").text = str(lon)
        ET.SubElement(location, "Altitude").text = str(altitude)
        ET.SubElement(location, "ChargingInfo").text = str(charging_info)
    tree = ET.ElementTree(root)
    return tree

# bounding box
top_left_lat = 52.525
top_left_lon = 8.409
bottom_right_lat = 49.013
bottom_right_lon = 13.369

# all data from api is xml
xml_data = query_osm_api(top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon)

print(xml_data)

# required data
extracted_data = extract_osm_xml(xml_data)

# output xml
xml_output = create_xml_output(extracted_data)


output_file = "POIs.xml"
xml_output.write(output_file, encoding="utf-8", xml_declaration=True)

print("OK", output_file)
