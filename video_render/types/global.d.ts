declare module 'parse-srt' {
	/**
	 * Represents a single subtitle item.
	 */
	export interface SubtitleItem {
	  id: number;
	  start: number;
	  end: number;
	  text: string;
	}

	/**
	 * Represents an array of subtitle items.
	 */
	export type Subtitles = SubtitleItem[];

	/**
	 * Parses an SRT (SubRip Subtitle) string and returns an array of subtitle items.
	 * @param srt - The SRT string to parse.
	 * @returns An array of subtitle items.
	 */
	export function parseSRT(srt: string): Subtitles;

	export default parseSRT;
  }
