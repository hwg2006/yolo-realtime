//! 基础数据类型：检测框、耗时统计与 COCO 类别表。

/// COCO 数据集类别数（YOLO11n 默认训练集）。
pub const NUM_CLASSES: usize = 80;

/// COCO 80 类名称，索引即类别 id。
pub const COCO_NAMES: [&str; NUM_CLASSES] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
];

/// 单个检测框（原图像素坐标）。
#[derive(Clone, Debug)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub cls: usize,
    pub conf: f32,
}

/// 单帧推理各阶段耗时（毫秒）。
#[derive(Default, Clone, Copy)]
pub struct Timings {
    pub pre: f64,
    pub infer: f64,
    pub post: f64,
}
