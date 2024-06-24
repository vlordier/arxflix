/* eslint-disable */
import type { PolarPathOptions } from "./types";
import { calculatePolarPosition } from './helpers';

/**
 * Generates a polar path based on the given frequency data and options.
 * @param frequenciesToDisplay - Array of frequency data to display.
 * @param options - Configuration options for the polar path.
 * @returns A string representing the SVG path.
 */
export function polarPath(frequenciesToDisplay: number[], options: PolarPathOptions): string {
  try {
    const {
      samples = frequenciesToDisplay.length,
      distance = 50,
      length = 100,
      top = 0,
      left = 0,
      type = "steps",
      startdeg = 0,
      enddeg = 360,
      invertdeg = false,
      invertpath = false,
      paths = [
        { d: "Q", sdeg: 0, sr: 0, deg: 50, r: 100, edeg: 100, er: 0 },
      ] as PolarPathOptions["paths"],
      normalizeFactor = 1,
    } = options;

    const normalizeData = frequenciesToDisplay.map((n) => n * normalizeFactor);
    let path = ``;
    const fixEndDeg = enddeg < startdeg ? enddeg + 360 : enddeg;
    const deg = !invertdeg ? (fixEndDeg - startdeg) / samples : (startdeg - fixEndDeg) / samples;
    const fixOrientation = !invertdeg ? 90 + startdeg : 90 + startdeg + 180;
    const invert = !invertpath ? 1 : -1;
    const pathslength = paths.length;
    const fixpathslength = type === "mirror" ? pathslength * 2 : pathslength;
    const pi180 = Math.PI / 180;

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
              const posX = calculatePolarPosition(
                deg,
                i,
                currentPath.sdeg,
                fixOrientation,
                length,
                currentPath.sr,
                normalizeDataValue,
                positive,
                mirror,
                invert,
                distance
              );
              const posY = calculatePolarPosition(
                deg,
                i,
                currentPath.sdeg,
                fixOrientation,
                length,
                currentPath.sr,
                normalizeDataValue,
                positive,
                mirror,
                invert,
                distance,
                true
              );

              const endPosX = calculatePolarPosition(
                deg,
                i,
                currentPath.edeg,
                fixOrientation,
                length,
                currentPath.er,
                normalizeDataValue,
                positive,
                mirror,
                invert,
                distance
              );
              const endPosY = calculatePolarPosition(
                deg,
                i,
                currentPath.edeg,
                fixOrientation,
                length,
                currentPath.er,
                normalizeDataValue,
                positive,
                mirror,
                invert,
                distance,
                true
              );

              if (posX !== lastPosX || posY !== lastPosY) {
                path += `M ${posX} ${posY} `;
              }

              path += `L ${endPosX} ${endPosY} `;

              lastPosX = endPosX;
              lastPosY = endPosY;
              break;
            }

            case "C": {
              const posX = calculatePolarPosition(
                deg,
                i,
                currentPath.sdeg,
                fixOrientation,
                length,
                currentPath.sr,
                normalizeDataValue,
                positive,
                mirror,
                invert,
                distance
              );
              const posY = calculatePolarPosition(
                deg,
                i,
                currentPath.sdeg,
                fixOrientation,
                length,
                currentPath.sr,
                normalizeDataValue,
                positive,
                mirror,
                invert,
                distance,
                true
              );

              const centerPosX = calculatePolarPosition(
                deg,
                i,
                currentPath.deg,
                fixOrientation,
                length,
                currentPath.r,
                normalizeDataValue,
                positive,
                mirror,
                invert,
                distance
              );
              const centerPosY = calculatePolarPosition(
                deg,
                i,
                currentPath.deg,
                fixOrientation,
                length,
                currentPath.r,
                normalizeDataValue,
                positive,
                mirror,
                invert,
                distance,
                true
              );

              const endPosX = calculatePolarPosition(
                deg,
                i,
                currentPath.edeg,
                fixOrientation,
                length,
                currentPath.er,
                normalizeDataValue,
                positive,
                mirror,
                invert,
                distance
              );
              const endPosY = calculatePolarPosition(
                deg,
                i,
                currentPath.edeg,
                fixOrientation,
                length,
                currentPath.er,
                normalizeDataValue,
                positive,
                mirror,
                invert,
                distance,
                true
              );

              if (posX !== lastPosX || posY !== lastPosY) {
                path += `M ${posX} ${posY} `;
              }

              path += `C ${posX} ${posY} ${centerPosX} ${centerPosY} ${endPosX} ${endPosY} `;

              lastPosX = endPosX;
              lastPosY = endPosY;
              break;
            }

            case "Q": {
              const posX = calculatePolarPosition(
                deg,
                i,
                currentPath.sdeg,
                fixOrientation,
                length,
                currentPath.sr,
                normalizeDataValue,
                positive,
                mirror,
                invert,
                distance
              );
              const posY = calculatePolarPosition(
                deg,
                i,
                currentPath.sdeg,
                fixOrientation,
                length,
                currentPath.sr,
                normalizeDataValue,
                positive,
                mirror,
                invert,
                distance,
                true
              );

              const centerPosX = calculatePolarPosition(
                deg,
                i,
                currentPath.deg,
                fixOrientation,
                length,
                currentPath.r,
                normalizeDataValue,
                positive,
                mirror,
                invert,
                distance
              );
              const centerPosY = calculatePolarPosition(
                deg,
                i,
                currentPath.deg,
                fixOrientation,
                length,
                currentPath.r,
                normalizeDataValue,
                positive,
                mirror,
                invert,
                distance,
                true
              );

              const endPosX = calculatePolarPosition(
                deg,
                i,
                currentPath.edeg,
                fixOrientation,
                length,
                currentPath.er,
                normalizeDataValue,
                positive,
                mirror,
                invert,
                distance
              );
              const endPosY = calculatePolarPosition(
                deg,
                i,
                currentPath.edeg,
                fixOrientation,
                length,
                currentPath.er,
                normalizeDataValue,
                positive,
                mirror,
                invert,
                distance,
                true
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
              const posX = calculatePolarPosition(
                deg,
                i,
                currentPath.sdeg,
                fixOrientation,
                length,
                currentPath.sr,
                normalizeDataValue,
                positive,
                mirror,
                invert,
                distance
              );
              const posY = calculatePolarPosition(
                deg,
                i,
                currentPath.sdeg,
                fixOrientation,
                length,
                currentPath.sr,
                normalizeDataValue,
                positive,
                mirror,
                invert,
                distance,
                true
              );

              const endPosX = calculatePolarPosition(
                deg,
                i,
                currentPath.edeg,
                fixOrientation,
                length,
                currentPath.er,
                normalizeDataValue,
                positive,
                mirror,
                invert,
                distance
              );
              const endPosY = calculatePolarPosition(
                deg,
                i,
                currentPath.edeg,
                fixOrientation,
                length,
                currentPath.er,
                normalizeDataValue,
                positive,
                mirror,
                invert,
                distance,
                true
              );

              if (posX !== lastPosX || posY !== lastPosY) {
                path += `M ${posX} ${posY} `;
              }

              const angle = (deg * i * currentPath.angle) / 100;
              const rx = (currentPath.rx * deg) / 100;
              const ry = (currentPath.ry * deg) / 100;

              let { sweep } = currentPath;
              if (positive === -1) {
                sweep = sweep === 1 ? 0 : 1;
              }
              if (mirror === -1) {
                sweep = sweep === 1 ? 0 : 1;
              }
              path += `A ${rx} ${ry} ${angle} ${currentPath.arc} ${sweep} ${endPosX} ${endPosY} `;

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
    console.error('Error generating polar path:', error);
    return '';
  }
}
