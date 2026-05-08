import MLX

public enum ImageLevels {
  public static func apply(image: MLXArray, min: Float, max: Float) -> MLXArray {
    guard min != max else {
      return MLXArray.zeros(like: image)
    }

    return MLX.clip((image - MLXArray(min)) / MLXArray(max - min), min: 0, max: 1)
  }

  public static func shouldApply(min: Float, max: Float) -> Bool {
    min != 0.0 || max != 1.0
  }
}
