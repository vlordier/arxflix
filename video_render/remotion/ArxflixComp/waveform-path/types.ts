/**
 * Options for linear path visualization
 */
export interface LinearPathOptions {
  samples?: number; // Number of samples for the path
  height?: number; // Height of the path
  width?: number; // Width of the path
  top?: number; // Top position of the path
  left?: number; // Left position of the path
  type?: "steps" | "mirror" | "bars"; // Type of the path visualization
  paths: LinearPath[]; // Array of linear paths
  normalizeFactor?: number; // Factor to normalize the path
  start?: number; // Start position
  end?: number; // End position
}

/**
 * Options for polar path visualization
 */
export interface PolarPathOptions {
  samples?: number; // Number of samples for the path
  distance?: number; // Distance for the polar path
  length?: number; // Length of the path
  top?: number; // Top position of the path
  left?: number; // Left position of the path
  type?: "steps" | "mirror" | "bars"; // Type of the path visualization
  startdeg?: number; // Start degree for the polar path
  enddeg?: number; // End degree for the polar path
  invertdeg?: boolean; // Invert degrees
  invertpath?: boolean; // Invert path
  paths: PolarPath[]; // Array of polar paths
  normalizeFactor?: number; // Factor to normalize the path
}

/**
 * Common path properties
 */
export interface Path {
  /**
   * Minimum show value, range 0 to 1
   * @default 0
   */
  minshow?: number;
  /**
   * Maximum show value, range 0 to 1
   * @default 1
   */
  maxshow?: number;
  /**
   * Normalize the path
   * @default false
   */
  normalize?: boolean;
}

/**
 * Linear line-to path
 */
export interface LinearLineToPath extends Path {
  d: "L";
  sx: number; // Start x
  sy: number; // Start y
  ex: number; // End x
  ey: number; // End y
}

/**
 * Polar line-to path
 */
export interface PolarLineToPath extends Path {
  d: "L";
  sdeg: number; // Start degree
  sr: number; // Start radius
  edeg: number; // End degree
  er: number; // End radius
}

/**
 * Horizontal path
 */
export interface HoritzontalPath extends Path {
  d: "H";
  sx: number; // Start x
  y: number; // y position
  ex: number; // End x
}

/**
 * Vertical path
 */
export interface VerticalPath extends Path {
  d: "V";
  sy: number; // Start y
  x: number; // x position
  ey: number; // End y
}

/**
 * Linear cubic Bezier curve path
 */
export interface LinearCubicBezierCurvePath extends Path {
  d: "C";
  sx: number; // Start x
  sy: number; // Start y
  x: number; // Control point x
  y: number; // Control point y
  ex: number; // End x
  ey: number; // End y
}

/**
 * Polar cubic Bezier curve path
 */
export interface PolarCubicBezierCurvePath extends Path {
  d: "C";
  sdeg: number; // Start degree
  sr: number; // Start radius
  deg: number; // Control point degree
  r: number; // Control point radius
  edeg: number; // End degree
  er: number; // End radius
}

/**
 * Linear quadratic Bezier curve path
 */
export interface LinearQuadraticBezierCurvePath extends Path {
  d: "Q";
  sx: number; // Start x
  sy: number; // Start y
  x: number; // Control point x
  y: number; // Control point y
  ex: number; // End x
  ey: number; // End y
}

/**
 * Polar quadratic Bezier curve path
 */
export interface PolarQuadraticBezierCurvePath extends Path {
  d: "Q";
  sdeg: number; // Start degree
  sr: number; // Start radius
  deg: number; // Control point degree
  r: number; // Control point radius
  edeg: number; // End degree
  er: number; // End radius
}

/**
 * Linear arc path
 */
export interface LinearArcPath extends Path {
  d: "A";
  sx: number; // Start x
  sy: number; // Start y
  ex: number; // End x
  ey: number; // End y
  rx: number; // Radius x
  ry: number; // Radius y
  angle: number; // Rotation angle
  arc: number; // Arc flag
  sweep: number; // Sweep flag
}

/**
 * Polar arc path
 */
export interface PolarArcPath extends Path {
  d: "A";
  sdeg: number; // Start degree
  sr: number; // Start radius
  edeg: number; // End degree
  er: number; // End radius
  rx: number; // Radius x
  ry: number; // Radius y
  angle: number; // Rotation angle
  arc: number; // Arc flag
  sweep: number; // Sweep flag
}

/**
 * Close path
 */
export interface ClosePath extends Path {
  d: "Z";
}

/**
 * Type for linear paths
 */
export type LinearPath =
  | LinearLineToPath
  | HoritzontalPath
  | VerticalPath
  | LinearCubicBezierCurvePath
  | LinearQuadraticBezierCurvePath
  | LinearArcPath
  | ClosePath;

/**
 * Type for polar paths
 */
export type PolarPath =
  | PolarLineToPath
  | PolarCubicBezierCurvePath
  | PolarQuadraticBezierCurvePath
  | PolarArcPath
  | ClosePath;
