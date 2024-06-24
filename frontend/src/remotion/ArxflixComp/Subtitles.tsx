import parseSRT, { SubtitleItem } from 'parse-srt';
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { makeTransform, translateY } from "@remotion/animation-utils";
import {
	cancelRender,
	continueRender,
	delayRender,
	useCurrentFrame,
	useVideoConfig,
} from 'remotion';
import { Word } from './Word';

// Custom hook to get windowed frame subtitles
const useWindowedFrameSubs = (
	src: string,
	options: { windowStart: number; windowEnd: number },
) => {
	const { windowStart, windowEnd } = options;
	const { fps } = useVideoConfig();

	// Parse SRT only if src is valid
	const parsed = useMemo(() => {
		if (!src) {
			console.warn('src is undefined or empty');
			return [];
		}
		try {
			return parseSRT(src);
		} catch (error) {
			console.error('Error parsing SRT:', error);
			return [];
		}
	}, [src]);

	// Filter and map subtitles to frames within the specified window
	return useMemo(() => {
		return parsed
			.map((item) => {
				const start = Math.floor(item.start * fps);
				const end = Math.floor(item.end * fps);
				return { item, start, end };
			})
			.filter(({ start }) => start >= windowStart && start <= windowEnd)
			.map<SubtitleItem>(({ item, start, end }) => ({
				...item,
				start,
				end,
			}));
	}, [fps, parsed, windowEnd, windowStart]);
};

// Component to render paginated subtitles
export const PaginatedSubtitles: React.FC<{
	subtitles: string;
	startFrame: number;
	endFrame: number;
	linesPerPage: number;
	subtitlesZoomMeasurerSize: number;
	subtitlesLineHeight: number;
	onlyDisplayCurrentSentence: boolean;
}> = ({
	startFrame,
	endFrame,
	subtitles,
	linesPerPage,
	subtitlesZoomMeasurerSize,
	subtitlesLineHeight,
	onlyDisplayCurrentSentence,
}) => {
	const frame = useCurrentFrame();
	const windowRef = useRef<HTMLDivElement>(null);
	const zoomMeasurer = useRef<HTMLDivElement>(null);
	const [handle] = useState(() => delayRender());
	const windowedFrameSubs = useWindowedFrameSubs(subtitles, {
		windowStart: startFrame,
		windowEnd: endFrame,
	});

	const [lineOffset, setLineOffset] = useState(0);

	const currentAndFollowingSentences = useMemo(() => {
		// Return all words if not restricted to current sentence
		if (!onlyDisplayCurrentSentence) return windowedFrameSubs;

		// Find the last index of the sentence ending before the current frame
		const indexOfCurrentSentence =
			windowedFrameSubs.findLastIndex((w, i) => {
				const nextWord = windowedFrameSubs[i + 1];
				return (
					nextWord &&
					(w.text.endsWith('?') ||
						w.text.endsWith('.') ||
						w.text.endsWith('!')) &&
					nextWord.start < frame
				);
			}) + 1;

		return windowedFrameSubs.slice(indexOfCurrentSentence);
	}, [frame, onlyDisplayCurrentSentence, windowedFrameSubs]);

	// Calculate the line offset based on the zoom and rendered lines
	useEffect(() => {
		const zoom =
			(zoomMeasurer.current?.getBoundingClientRect().height as number) /
			subtitlesZoomMeasurerSize;
		const linesRendered =
			(windowRef.current?.getBoundingClientRect().height as number) /
			(subtitlesLineHeight * zoom);
		const linesToOffset = Math.max(0, linesRendered - linesPerPage);
		setLineOffset(linesToOffset);
		continueRender(handle);
	}, [
		frame,
		handle,
		linesPerPage,
		subtitlesLineHeight,
		subtitlesZoomMeasurerSize,
	]);

	const currentFrameSentences = currentAndFollowingSentences.filter((word) => word.start < frame);

	return (
		<div className="relative overflow-hidden px-10">
			<div
				ref={windowRef}
				style={{
					transform: makeTransform([translateY(-lineOffset * subtitlesLineHeight)]),
				}}
			>
				{currentFrameSentences.map((item) => (
					<span key={item.id} id={String(item.id)}>
						{/* Add space before the word if it doesn't start with '-' or a space */}
						{item.text.startsWith('-') || item.text.startsWith(' ') ? '' : ' '}
						<Word frame={frame} item={item} />
					</span>
				))}
			</div>
			<div
				ref={zoomMeasurer}
				style={{
					height: subtitlesZoomMeasurerSize,
					width: subtitlesZoomMeasurerSize,
				}}
			/>
		</div>
	);
};

// TypeScript declaration for findLastIndex
declare global {
	interface Array<T> {
		findLastIndex(
			predicate: (value: T, index: number, obj: T[]) => unknown,
			thisArg?: unknown,
		): number;
	}
}
