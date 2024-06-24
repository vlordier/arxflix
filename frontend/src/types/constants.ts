import { zColor } from "@remotion/zod-types";
import { staticFile } from "remotion";
import { z } from "zod";

// Constants
export const COMP_NAME = "MyComp";
const AUDIO_FILE_EXTENSION = '.mp3';
const SRT_FILE_EXTENSION = '.srt';
const JSON_FILE_EXTENSION = '.json';

// Schema for composition properties using Zod for validation
export const CompositionProps = z.object({
  durationInSeconds: z.number().positive(), // Positive number for video duration
  audioOffsetInSeconds: z.number().min(0), // Non-negative number for audio offset
  subtitlesFileName: z.string().refine((s) => s.endsWith(SRT_FILE_EXTENSION), {
    message: `Subtitles file must be a ${SRT_FILE_EXTENSION} file`, // Ensure subtitle file ends with .srt
  }),
  audioFileName: z.string().refine((s) => s.endsWith(AUDIO_FILE_EXTENSION), {
    message: `Audio file must be a ${AUDIO_FILE_EXTENSION} file`, // Ensure audio file ends with .mp3
  }),
  richContentFileName: z.string().refine((s) => s.endsWith(JSON_FILE_EXTENSION), {
    message: `Rich content file must be a ${JSON_FILE_EXTENSION} file`, // Ensure rich content file ends with .json
  }),
  waveColor: zColor(), // Color for wave representation
  subtitlesLinePerPage: z.number().int().min(0), // Non-negative integer for lines per page
  subtitlesLineHeight: z.number().int().min(0), // Non-negative integer for line height
  subtitlesZoomMeasurerSize: z.number().int().min(0), // Non-negative integer for zoom measurer size
  onlyDisplayCurrentSentence: z.boolean(), // Boolean to display only current sentence
  mirrorWave: z.boolean(), // Boolean to mirror wave
  waveLinesToDisplay: z.number().int().min(0), // Non-negative integer for wave lines to display
  waveFreqRangeStartIndex: z.number().int().min(0), // Non-negative integer for wave frequency range start index
  waveNumberOfSamples: z.enum(['32', '64', '128', '256', '512']), // Enum for number of samples
});

// Type for CompositionProps inferred from the schema
export type CompositionPropsType = z.infer<typeof CompositionProps>;

// Default properties for the composition
export const defaultCompositionProps: CompositionPropsType = {
  // Audio settings
  audioOffsetInSeconds: 0,
  audioFileName: staticFile('audio.wav'), // Default audio file

  // Rich content settings
  richContentFileName: staticFile('output.json'), // Default rich content file

  // Subtitles settings
  subtitlesFileName: staticFile('output.srt'), // Default subtitles file
  onlyDisplayCurrentSentence: true,
  subtitlesLinePerPage: 2,
  subtitlesZoomMeasurerSize: 10,
  subtitlesLineHeight: 98,

  // Wave settings
  waveColor: '#a3a5ae', // Default wave color
  waveFreqRangeStartIndex: 5,
  waveLinesToDisplay: 300,
  waveNumberOfSamples: '512', // Default number of samples (string for Remotion controls)
  mirrorWave: false,

  // Metadata settings
  durationInSeconds: 5, // Default duration in seconds
};

// Video configuration constants
export const DURATION_IN_FRAMES = 200; // Duration of the video in frames
export const VIDEO_WIDTH = 1920; // Width of the video
export const VIDEO_HEIGHT = 1080; // Height of the video
export const VIDEO_FPS = 30; // Frames per second for the video
