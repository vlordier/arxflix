/* eslint-disable */
import type { LinearPathOptions } from "./types";
import { calculateHeight, calculatePosition } from './helpers';

/**
 * Generates a linear path based on the given frequency data and options.
 * @param frequenciesToDisplay - Array of frequency data to display.
 * @param options - Configuration options for the linear path.
 * @returns A string representing the SVG path.
 */
export function linearPath(frequenciesToDisplay: number[], options: LinearPathOptions): string {
  try {
    const {
      samples = frequenciesToDisplay.length,
      normalizeFactor = 1,
      height = 100,
      width = 800,
      top = 0,
      left = 0,
      type = "steps",
      paths = [
        { d: "Q", sx: 0, sy: 0, x: 50, y: 100, ex: 100, ey: 0 },
      ] as LinearPathOptions["paths"],
    } = options;

    const normalizeData = frequenciesToDisplay.map((n) => n * normalizeFactor);
    let path = ``;

    const fixHeight = type !== "bars" ? (height + top * 2) / 2 : height + top;
    const fixWidth = width / samples;
    const pathslength = paths.length;
    const fixpathslength = type === "mirror" ? pathslength * 2 : pathslength;

    let lastPosX = -9999;
    let lastPosY = -9999;

    for (let i = 0; i < samples; i++) {
      const positive = type !== "bars" ? (i % 2 ? 1 : -1) : 1;
      let mirror = 1;
      for (let j = 0; j < fixpathslength; j++) {
        let k = j;
        if (j >= pathslength) {
          k = j - pathslength;
          mirror = -1;
        }
        const currentPath = paths[k];
        const normalizeDataValue = currentPath.normalize ? 1 : normalizeData[i];
        currentPath.minshow = currentPath.minshow ?? 0;
        currentPath.maxshow = currentPath.maxshow ?? 1;
        currentPath.normalize = currentPath.normalize ?? false;

        if (currentPath.minshow <= normalizeData[i] && currentPath.maxshow >= normalizeData[i]) {
          switch (currentPath.d) {
            case "L": {
              const posX = calculatePosition(i, fixWidth, currentPath.sx, left);
              const posY = calculateHeight(
                fixHeight,
                normalizeDataValue,
                currentPath.sy,
                height,
                positive,
                mirror,
                type
              );

              const endPosX = calculatePosition(i, fixWidth, currentPath.ex, left);
              const endPosY = calculateHeight(
                fixHeight,
                normalizeDataValue,
                currentPath.ey,
                height,
                positive,
                mirror,
                type
              );

              if (posX !== lastPosX || posY !== lastPosY) {
                path += `M ${posX} ${posY} `;
              }

              path += `L ${endPosX} ${endPosY} `;

              lastPosX = endPosX;
              lastPosY = endPosY;
              break;
            }

            case "H": {
              const posX = calculatePosition(i, fixWidth, currentPath.sx, left);
              const posY = calculateHeight(
                fixHeight,
                normalizeDataValue,
                currentPath.y,
                height,
                positive,
                mirror,
                type
              );

              const endPosX = calculatePosition(i, fixWidth, currentPath.ex, left);

              if (posX !== lastPosX || posY !== lastPosY) {
                path += `M ${posX} ${posY} `;
              }

              path += `H ${endPosX} `;

              lastPosX = endPosX;
              lastPosY = posY;
              break;
            }

            case "V": {
              const posX = calculatePosition(i, fixWidth, currentPath.x, left);
              const posY = calculateHeight(
                fixHeight,
                normalizeDataValue,
                currentPath.sy,
                height,
                positive,
                mirror,
                type
              );

              const endPosY = calculateHeight(
                fixHeight,
                normalizeDataValue,
                currentPath.ey,
                height,
                positive,
                mirror,
                type
              );

              if (posX !== lastPosX || posY !== lastPosY) {
                path += `M ${posX} ${posY} `;
              }

              path += `V ${endPosY} `;

              lastPosX = posX;
              lastPosY = endPosY;
              break;
            }

            case "C": {
              const posX = calculatePosition(i, fixWidth, currentPath.sx, left);
              const posY = fixHeight - ((fixHeight * currentPath.sy) / 100) * positive;

              const centerPosX = calculatePosition(i, fixWidth, currentPath.x, left);
              const centerPosY = calculateHeight(
                fixHeight,
                normalizeDataValue,
                currentPath.y,
                height,
                positive,
                mirror,
                type
              );

              const endPosX = calculatePosition(i, fixWidth, currentPath.ex, left);
              const endPosY = fixHeight - ((fixHeight * currentPath.ey) / 100) * positive;

              if (posX !== lastPosX || posY !== lastPosY) {
                path += `M ${posX} ${posY} `;
              }

              path += `C ${posX} ${posY} ${centerPosX} ${centerPosY} ${endPosX} ${endPosY} `;

              lastPosX = endPosX;
              lastPosY = endPosY;
              break;
            }

            case "Q": {
              const posX = calculatePosition(i, fixWidth, currentPath.sx, left);
              const posY = calculateHeight(
                fixHeight,
                normalizeDataValue,
                currentPath.sy,
                height,
                positive,
                mirror,
                type
              );

              const centerPosX = calculatePosition(i, fixWidth, currentPath.x, left);
              const centerPosY = calculateHeight(
                fixHeight,
                normalizeDataValue,
                currentPath.y,
                height,
                positive,
                mirror,
                type
              );

              const endPosX = calculatePosition(i, fixWidth, currentPath.ex, left);
              const endPosY = calculateHeight(
                fixHeight,
                normalizeDataValue,
                currentPath.ey,
                height,
                positive,
                mirror,
                type
              );

              if (posX !== lastPosX || posY !== lastPosY) {
                path += `M ${posX} ${posY} `;
              }

              path += `Q ${centerPosX} ${centerPosY} ${endPosX} ${endPosY} `;

              lastPosX = endPosX;
              lastPosY = endPosY;
              break;
            }

            case "A": {
              const posX = calculatePosition(i, fixWidth, currentPath.sx, left);
              const posY = calculateHeight(
                fixHeight,
                normalizeDataValue,
                currentPath.sy,
                height,
                positive,
                mirror,
                type
              );

              const endPosX = calculatePosition(i, fixWidth, currentPath.ex, left);
              const endPosY = calculateHeight(
                fixHeight,
                normalizeDataValue,
                currentPath.ey,
                height,
                positive,
                mirror,
                type
              );

              if (posX !== lastPosX || posY !== lastPosY) {
                path += `M ${posX} ${posY} `;
              }

              const rx = (currentPath.rx * fixWidth) / 100;
              const ry = (currentPath.ry * fixWidth) / 100;
              let { sweep } = currentPath;
              if (positive === -1) {
                sweep = sweep === 1 ? 0 : 1;
              }
              if (mirror === -1) {
                sweep = sweep === 1 ? 0 : 1;
              }
              path += `A ${rx} ${ry} ${currentPath.angle} ${currentPath.arc} ${sweep} ${endPosX} ${endPosY} `;

              lastPosX = endPosX;
              lastPosY = endPosY;
              break;
            }

            case "Z":
              path += "Z ";
              break;

            default:
              break;
          }
        }
      }
    }
    return path;
  } catch (error) {
    console.error('Error generating linear path:', error);
    return '';
  }
}
