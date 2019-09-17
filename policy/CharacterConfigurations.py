# motion = "walk_extension"
# motion = "basketball"
# motion = "zombie"
# motion = "gorilla"
# motion = "walkrun"
# motion = "walk_new"
# motion = "walkfall"
motion = "walkrunfall"
#motion = "walkfall_variation"
# motion = "punchfall"
# motion = "walkfall_prediction_h0.2"
# motion = "jog_roll"

ALL_JOINTS = False
CMU_JOINTS = False
NEW_JOINTS = False

if motion == "walk_extension":
	ALL_JOINTS = True
elif motion == "walkrun":
	CMU_JOINTS = True
elif motion == "basketball":
	ALL_JOINTS = True
elif motion == "zombie":
	CMU_JOINTS = True
elif motion == "gorilla":
	CMU_JOINTS = True
elif motion == "walk_new":
	NEW_JOINTS = True
elif motion == "walkfall":
	NEW_JOINTS = True
elif motion.startswith("walkfall_prediction"):
	NEW_JOINTS = True
elif motion == "walkfall_variation":
	NEW_JOINTS = True
elif motion == "jog_roll":
	NEW_JOINTS = True
elif motion == "punchfall":
	NEW_JOINTS = True
elif motion == "walkrunfall":
	NEW_JOINTS = True

if(ALL_JOINTS):
	INPUT_MOTION_SIZE = 72
	FOOT_CONTACT_OFFSET = 70	
	OUTPUT_MOTION_SIZE = 72

if(CMU_JOINTS):
	INPUT_MOTION_SIZE = 78
	FOOT_CONTACT_OFFSET = 70	
	OUTPUT_MOTION_SIZE = 78

if(NEW_JOINTS):
	INPUT_MOTION_SIZE = 54
	FOOT_CONTACT_OFFSET = 52
	OUTPUT_MOTION_SIZE = 111

FEEDBACK_INTERVAL = 1

USE_EPI_LEN = False
USE_EXP_WEIGHT = True
USE_MCHT= False
