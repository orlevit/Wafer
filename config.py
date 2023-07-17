import os

# Locations
DATA_BASE_DIR = r'C:\Users\or\PycharmProjects\annotator_api\notebooks\try\wafer'  # Base location
TRAIN_DATA_LOC = os.path.join(DATA_BASE_DIR, 'data', 'df_wafers.csv')  # Training dataframe location
TEST_DATA_LOC = os.path.join(DATA_BASE_DIR, 'data', 'df_wafers_test.csv')  # Testing dataframe location
IMAGES_SAVE_LOC = os.path.join(DATA_BASE_DIR, 'output_images')  # Output predicted scratch location

# Hyper-parameters
POLY_ORDER = range(2, 5)  # Order of the polynom fit
IF_SAVE_IMAGES = False  # Indication whether to save the image
MIN_SEG_AREA_ALLOWED = 3  # The minimum area of the considered connected component
SCORE_STRENGTH_OF_INK_PIXELS = [-0.7, -0.5, -0.3]  # The wighted score of inked pixels. recommended range: [-1,0]
RANDOM_STATE = 42  # Global random state
WAFER_SAMPLE_SIZE = 1000  # The size of the unique wafers to search through the Hyper-parameters
GOOD_DIES_RAIO = 0.88  # The ratio of good dies to total dies. below this ratio no scratch is computed
COMBINATION_SIZE_LIMIT = 10  # Number of segments, should be small as there is a loop pn all the possible combinations
TRAIN_IND = 'Train'  # If this is training process
TEST_IND = 'Test'  # If this is Testing process
MIDDLE_LEFT_WAFER_SIZE_RANGE = 35  # Left range of the middle wafer size. For scratches filtered by size
MIDDLE_RIGHT_WAFER_SIZE_RANGE = 63  # Right range of the middle wafer size. For scratches filtered by size
RIGHT_WAFER_SIZE_RANGE = 105  # Right wafer size boundary.
