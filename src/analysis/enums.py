from enum import Enum, IntEnum

class Measurements(Enum):
    EYEBROW_EYE_DISTANCE = 2
    EYEBROW_HORIZONTAL_DISTANCE = 3
    EYEBROW_INTERCEPT_DISTANCE = 4
    MOUTH_AREA = 5
    LIPCENTER_OFFSET = 6
    EYE_DROOP = 7
    NASOLABIAL_FOLD = 8

class FlaccidCategories(Enum):
    CompleteFlaccid = 1
    SevereFlaccid = 2
    ModerateFlaccid = 3
    MildFlaccid = 4
    NearNormalFlaccid = 5
    Normal = 6

class MEEIMovements(IntEnum):
    RELAXED = 0
    EYEBROWS = 1
    EYE_CLOSE_SOFT = 2
    EYE_CLOSE_HARD = 3
    SMILE_CLOSED = 4
    SMILE_OPEN = 5
    LIP_PUCKER = 6
    BOTTOM_TEETH = 7