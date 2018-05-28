from urdf_parser_py.urdf import URDF
robot = URDF.from_xml_string("<robot name='myrobot'></robot>")
print(robot)