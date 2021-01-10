from math import radians, degrees
import bpy
import sys

# Reset Blend
imageNum = 0

# Create Folder Path
folderPath = ""

# create global variables
context = None
scene = None
robot = None
fastRender = False


# Renders the Scene
def render():
    global imageNum
    global folderPath
    scene.camera = bpy.data.objects['RobotCamera']
    bpy.context.scene.render.filepath = folderPath+"/OutputImages/"+str(imageNum)
    bpy.ops.render.render(write_still=True)
    scene.camera = bpy.data.objects['Camera']
    bpy.context.scene.render.filepath = folderPath+"/OverheadImages/"+str(imageNum)
    bpy.ops.render.render(write_still=True)
    imageNum += 1


# Rotates ROBs position
def rotate(deg):
    if fastRender:
        n = 45
    else:
        n = 5
    if deg > 0:
        for i in range(0, deg, n):
            robot.rotation_euler[2] += radians(n)
            render()
    else:
        deg = -deg
        for i in range(0, deg, n):
            robot.rotation_euler[2] += radians(-n)
            render()


# Moves ROB a set distance
def forward(distance):
    global imageNum
    if distance == 0:
        render()
    deg = round(degrees(robot.rotation_euler[2]))
    if deg % 360 == 0 or deg == 0:
        robot.location[1] += distance
    elif deg % 270 == 0:
        robot.location[0] += distance
    elif deg % 180 == 0:
        robot.location[1] -= distance
    else:
        robot.location[0] -= distance
    render()


# Sets up the Blender environment and executes the transversal
def main():
    global imageNum
    global folderPath
    global context
    global scene
    global robot
    global fastRender

    # Initialize Blender File
    bpy.ops.wm.open_mainfile(filepath="BlenderFiles/"+str(sys.argv[12]))

    # Initialize scene and set ROB as the active object
    context = bpy.context
    scene = context.scene
    bpy.data.objects['ROB'].select_set(True)
    fastRender = ("True" == sys.argv[13])
    robot = bpy.context.active_object

    # Set the Image Number for Render Output
    imageNum = int(sys.argv[4])

    # Place Robot in Correct Position within the scene
    robot.location[0], robot.location[1], robot.location[2], robot.rotation_euler[2] = float(sys.argv[5]), float(
        sys.argv[6]), float(sys.argv[7]), float(sys.argv[8])

    # Sets the location of the output images
    folderPath = str(sys.argv[11])

    # Move ROB to Next Position
    forward(float(sys.argv[9]))
    rotate(int(sys.argv[10]))

    # Write ROB's final position to txt file
    f = open("position.txt", "w")
    f.write(
        str(robot.location[0])+" "+str(robot.location[1])+" "+str(robot.location[2])+" "+str(robot.rotation_euler[2]))
    f.close()


if __name__ == "__main__":
    main()
