from enum import Enum, IntEnum

class Measurements(Enum):
    Eyebrows = 1,
    EYEBROW_EYE_DISTANCE = 2,
    EYEBROW_HORIZONTAL_DISTANCE = 3,
    EYEBROW_INTERCEPT_DISTANCE = 4,
    MOUTH_AREA = 5,
    LIPCENTER_OFFSET = 6,
    EYE_DROOP = 7

class FlaccidCategories(Enum):
    CompleteFlaccid = 1
    SevereFlaccid = 2
    ModerateFlaccid = 3
    MildFlaccid = 4
    NearNormalFlaccid = 5
    Normal = 6