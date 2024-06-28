/**
 * Helper function to calculate the position in linear paths.
 */
export function calculatePosition(index: number, fixWidth: number, percentage: number, offset: number): number {
    return index * fixWidth + (fixWidth * percentage) / 100 + offset;
  }

  /**
   * Helper function to calculate the height in linear paths.
   */
  export function calculateHeight(
    fixHeight: number,
    normalizeDataValue: number,
    percentage: number,
    height: number,
    positive: number,
    mirror: number,
    type: string
  ): number {
    return (
      fixHeight +
      ((normalizeDataValue * percentage) / 100) * (type !== "bars" ? height / 2 : height) * -positive * mirror
    );
  }

  /**
   * Helper function to calculate the position in polar paths.
   */
  export function calculatePolarPosition(
    deg: number,
    index: number,
    percentage: number,
    fixOrientation: number,
    length: number,
    radiusPercentage: number,
    normalizeDataValue: number,
    positive: number,
    mirror: number,
    invert: number,
    distance: number,
    isY: boolean = false
  ): number {
    const angle = (deg * (index + percentage / 100) - fixOrientation) * (Math.PI / 180);
    return isY
      ? length * (radiusPercentage / 100) * normalizeDataValue * positive * mirror * invert + distance * Math.sin(angle)
      : length * (radiusPercentage / 100) * normalizeDataValue * positive * mirror * invert + distance * Math.cos(angle);
  }
