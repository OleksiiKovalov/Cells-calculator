from model.tracker import Tracker
from model.utils import COLOR_NUMBER as color_number
object_size = { 
                'min_size' : 0.0,
                   'max_size' : 100,
                   'signal' : print,
                   'round_parametr_slider' : 10**6,
                   'round_parametr_value_input' : 10**4,
                   'color_map' : "virdis",
                   'color_map_list' : list(color_number.keys()),
                   'line_width' : 100.00

                   
        }
tracker = Tracker("model/yolov8n-seg.pt", object_size)

tracker.track("C:\уник\\tracker imeges")
