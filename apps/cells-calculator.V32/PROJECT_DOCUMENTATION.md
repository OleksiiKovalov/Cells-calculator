# Cells Calculator Project - Architecture Documentation

## 1. Project Overview

**Cells Calculator** is a comprehensive PyQt5-based application designed for automated cell detection, counting, and analysis in microscopy images. The application supports multiple detection algorithms, provides an intuitive user interface for image visualization, and implements a modular plugin architecture for extensibility.

### Key Features:
- Multi-algorithm cell detection (YOLO, Cellpose, InstanSeg, Stardist)
- Interactive image visualization with zoom and pan capabilities
- Batch processing capabilities
- Plugin-based architecture for easy extensibility
- Support for various image formats (JPG, PNG, TIF, BMP, LSM)
- Real-time parameter adjustment with auto-apply functionality

---

## 2. Application Entry Point

### main.py
**Purpose**: Application initialization and launch point

**Key Components**:
- **Splash Screen**: Displays loading progress during startup
- **Model Registration**: Loads and registers available detection models
- **Progress Tracking**: Provides user feedback during initialization

**Architecture Flow**:
```
main.py → SplashScreen → Model Loading → MainWindow Initialization → Application Ready
```

**Critical Code Sections**:
```python
known_models = {
    "YOLO v8": {"path": 'trainedmodels/yolov8m-det.onnx', "size": object_size},
    "YOLO 11x 512": {"path": 'trainedmodels/YOLO11x-512-seg.pt', "size": object_size},
    # ... additional models
}
```

---

## 3. Main Application Window

### MainWindow.py
**Purpose**: Primary application interface and layout manager

**Architecture Components**:

#### 3.1 Window Structure
```
MainWindow (QMainWindow)
├── MenuBar (menubar)
├── Central Widget (QWidget)
│   └── Main Splitter (QSplitter - Horizontal)
│       ├── Main Graphics View (75% width)
│       │   └── Graphics Scene (Image Display)
│       └── Right Layout Widget (25% width)
│           └── Plugin Area (Dynamic)
└── Status Bar (Multi-section status display)
```

#### 3.2 Core Functionality
- **Image Display System**: QGraphicsView/QGraphicsScene for scalable image rendering
- **Zoom/Pan Controls**: Mouse wheel zoom, drag-to-pan functionality
- **Plugin Management**: Dynamic loading and switching between plugins
- **Signal Communication**: PyQt signal-slot pattern for component interaction

#### 3.3 Key Methods
- `init_mainScene()`: Sets up graphics view and splitter layout
- `add_image()`: Loads and displays images in the main view
- `handle_menubar_action()`: Processes menu selections
- `handle_rightLayout_action()`: Manages plugin communications

---

## 4. Business Logic Layer

### 4.1 BaseModel.py
**Purpose**: Abstract foundation for all detection algorithms

**Class Hierarchy**:
```
BaseModel (Abstract Base Class)
├── CellCounter (ONNX-based detection)
├── InstanSegSegmenter (InstanSeg implementation)  
├── CellposeSegmenter (Cellpose integration)
├── StardistSegmenter (Stardist implementation)
└── YOLOSegmenter (YOLO-based segmentation)
```

**Abstract Interface**:
```python
class BaseModel:
    def init_models(self):           # Model initialization
    def init_x10_model(self, path):  # Low magnification model
    def init_x20_model(self, path):  # High magnification model
    def count_x10(self, image, filename):  # Low mag processing
    def count_x20(self, image, filename):  # High mag processing
```

### 4.2 Model.py
**Purpose**: Unified model interface and factory

**Functionality**:
- **Dynamic Model Loading**: Uses importlib for runtime model instantiation
- **Model Registration**: Links model types to implementation classes
- **Unified Interface**: Provides consistent API across different algorithms
- **Error Handling**: Robust error management for model initialization failures

**Key Components**:
```python
def init_counter(self, path, object_size, model_type, model_data=None):
    cell_counter_class_name = get_registered_model(model_type).get('model_class')
    # Dynamic class loading and instantiation
```

---

## 5. Plugin Architecture

### 5.1 BasePlugin.py
**Purpose**: Foundation for all application plugins

**Plugin Interface**:
```python
class BasePlugin(QObject):
    plugin_signal = pyqtSignal(str, object)
    
    def get_name(self):          # Plugin identification
    def init_value(self):        # Parameter initialization  
    def init_rightLayout(self):  # UI layout creation
    def handle_action(self, action_name, value):  # Action processing
```

### 5.2 Available Plugins

#### CellDetectorPlugin.py
**Primary Function**: Cell detection and analysis
**UI Components**:
- Model selection dropdown
- Range sliders with auto-apply functionality
- Display mode radio buttons (Original/Inference/Detections)
- Batch processing controls
- Results visualization area
- Colormap and transparency controls

#### TrackerPlugin.py
**Primary Function**: Cell tracking across time series
**UI Components**:
- Model selection for tracking algorithms
- Results display area
- Processing controls

#### SpheroidSegmenterPlugin.py
**Primary Function**: Spheroid segmentation and analysis
**UI Components**:
- Segmentation model selection
- Object size controls
- Visualization toggles

### 5.3 Plugin Loading System
```
MainWindow
├── init_value() → Plugin Dictionary Creation
├── right_layout() → Plugin Container
│   ├── init_rightLayout() → Active Plugin Loading
│   └── handle_plugin_signal() → Plugin Communication
└── Plugin Signal Routing → UI Updates
```

---

## 6. User Interface Hierarchy

### 6.1 Layout Structure
```
MainWindow
├── MenuBar
│   ├── File Menu (Open, Save, Export)
│   ├── Edit Menu (Settings, Preferences)
│   ├── View Menu (Zoom, Display Options)
│   └── Plugin Menu (Switch between plugins)
├── Central Area (QSplitter)
│   ├── Main Graphics View (Image Display)
│   │   ├── Zoom Controls (Ctrl+Wheel)
│   │   ├── Pan Controls (Mouse Drag)
│   │   └── Image Overlay (Detection Results)
│   └── Right Panel (Plugin UI)
│       ├── Plugin Selection
│       ├── Parameter Controls
│       ├── Processing Buttons
│       └── Results Display
└── Status Bar
    ├── Main Status
    ├── File Information
    └── Processing Status
```

### 6.2 Signal-Slot Communication Pattern
```
MenuBar ←→ MainWindow ←→ RightLayout ←→ ActivePlugin
   ↓           ↓              ↓            ↓
Actions    Coordination   Routing    Implementation
```

---

## 7. Component Interactions and Data Flow

### 7.1 Application Startup Sequence
1. **main.py**: Initialize splash screen and progress tracking
2. **Model Registration**: Load available models from configuration
3. **MainWindow Creation**: Initialize UI components and layouts
4. **Plugin System Setup**: Create plugin dictionary and load default plugin
5. **Signal Connection**: Establish communication pathways
6. **UI Finalization**: Complete layout setup and show window

### 7.2 Image Processing Workflow
```
Image Loading (MainWindow)
    ↓
Display in Graphics View
    ↓
Plugin Selection (User Choice)
    ↓
Parameter Configuration (Plugin UI)
    ↓
Model Execution (BaseModel Implementation)
    ↓
Result Processing (Plugin)
    ↓
Visualization Update (Graphics View Overlay)
```

### 7.3 Plugin Communication Flow
```
User Interaction (Plugin UI)
    ↓
Plugin Event Handler
    ↓
plugin_signal.emit(action, data)
    ↓
right_layout.handel_plugin_signal()
    ↓
MainWindow.handle_rightLayout_action()
    ↓
UI Updates / Model Execution
```

---

## 8. Advanced UI Components

### 8.1 Range Slider System
**File**: `RangeSliderWrapper` in CellDetectorPlugin.py
**Features**:
- **Auto-Apply Mode**: Immediate parameter updates
- **Manual Apply Mode**: Explicit user confirmation
- **Visual Feedback**: Real-time value display
- **Lock Mechanism**: Prevents recursive updates

### 8.2 Graphics Display System
**Components**:
- **Main Scene**: Primary image display with zoom/pan
- **Overlay System**: Detection result visualization
- **Multi-mode Display**: Original, Inference, Detection views

---

## 9. Model Integration Patterns

### 9.1 Model Loading Strategy
- **Dynamic Import**: Runtime class loading based on configuration
- **Type Registration**: Central registry for available models
- **Error Isolation**: Graceful handling of missing models
- **Resource Management**: Efficient memory usage for large models

### 9.2 Detection Pipeline
```
Image Input → Preprocessing → Model Inference → Post-processing → Visualization
     ↓              ↓              ↓              ↓              ↓
Format Check   Resizing     Algorithm      Filtering      Overlay
Channel Ext    Normalization  Execution    Confidence      Display
Quality Val    Augmentation   Batching     NMS Filter      Export
```

---

## 10. Key Design Patterns

### 10.1 Plugin Pattern
- **BasePlugin**: Abstract interface for all plugins
- **Dynamic Loading**: Runtime plugin registration and switching
- **Isolation**: Independent plugin development and testing

### 10.2 Observer Pattern
- **Signal-Slot Communication**: Decoupled component interaction
- **Event Broadcasting**: Multi-subscriber notifications
- **State Synchronization**: Consistent UI updates

### 10.3 Factory Pattern
- **Model Creation**: Dynamic model instantiation
- **Plugin Instantiation**: Runtime plugin creation
- **Configuration-Driven**: Settings-based object creation

---

## 11. File and Directory Organization

### 11.1 Core Application Structure
```
├── main.py                          # Application entry point
├── UI/
│   ├── MainWindow.py                # Main application window
│   ├── right_layout/
│   │   ├── right_layout.py          # Plugin container
│   │   └── plugins/
│   │       ├── BasePlugin.py        # Plugin base class
│   │       ├── CellDetectorPlugin.py # Primary detection plugin
│   │       ├── TrackerPlugin.py     # Cell tracking plugin
│   │       └── SpheroidSegmenterPlugin.py # Spheroid analysis
│   └── [other UI components...]
├── model/
│   ├── BaseModel.py                 # Model base class
│   ├── Model.py                     # Model factory
│   ├── CellCounter.py               # ONNX-based detection
│   ├── InstanSegSegmenter.py        # InstanSeg implementation
│   ├── CellposeSegmenter.py         # Cellpose integration
│   └── [other models...]
└── [data directories...]
```

---

## 12. Extension Points and Customization

### 12.1 Adding New Models
1. **Inherit from BaseModel**: Implement required abstract methods
2. **Register Model Type**: Add to model registry in configuration
3. **Update Plugin**: Add model to plugin selection options

### 12.2 Creating New Plugins
1. **Inherit from BasePlugin**: Implement plugin interface
2. **Define UI Layout**: Create plugin-specific interface
3. **Register Plugin**: Add to plugin dictionary in MainWindow
4. **Handle Events**: Implement action processing methods

### 12.3 UI Customization
- **Theme Support**: Modify CSS stylesheets
- **Layout Adjustments**: Customize splitter ratios and widget sizing
- **Control Extensions**: Add new parameter controls to plugin interfaces

---

## 13. Performance and Architecture Considerations

### 13.1 Memory Management
- **Lazy Loading**: Models loaded on demand
- **Image Caching**: Efficient image data handling
- **Resource Cleanup**: Proper object disposal in plugins

### 13.2 Responsiveness
- **Background Processing**: Non-blocking model execution
- **Progress Feedback**: User-visible processing status
- **Threaded Operations**: Separate processing from UI thread

### 13.3 Scalability
- **Plugin Architecture**: Easy feature additions
- **Modular Design**: Independent component development
- **Configuration-Driven**: Flexible behavior modification

