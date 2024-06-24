import React from 'react';
import {
	Img,
	interpolate,
	useCurrentFrame,
	useVideoConfig,
} from 'remotion';
import { InlineMath } from 'react-katex';

// Type definition for rich content
export type RichContent = {
	type: 'figure' | 'headline' | 'equation';
	content: string;
	start: number;
	end: number;
};

// Component to display the current figure, headline, or equation based on the current frame
export const CurrentFigure: React.FC<{
	richContent: RichContent[];
	transitionFrames?: number;
}> = ({ richContent = [], transitionFrames = 10 }) => {
	const frame = useCurrentFrame();
	const { fps } = useVideoConfig();

	// Find the current figure based on the frame and the start/end times
	const currentFigure = richContent.find(
		(f) => frame >= (f.start * fps) && frame <= (f.end * fps)
	);

	// Return null if no current figure is found
	if (!currentFigure) {
		return null;
	}

	const currentFigureDurationInFrame = (currentFigure.end * fps) - (currentFigure.start * fps);

	// Calculate scale and opacity for transition effects
	const scale = interpolate(
		frame - (currentFigure.start * fps),
		[0, transitionFrames, currentFigureDurationInFrame - transitionFrames, currentFigureDurationInFrame],
		[0.5, 1, 1, 0.5],
		{ extrapolateRight: 'clamp' }
	);

	const opacity = interpolate(
		frame - (currentFigure.start * fps),
		[0, transitionFrames, currentFigureDurationInFrame - transitionFrames, currentFigureDurationInFrame],
		[0, 1, 1, 0],
		{ extrapolateRight: 'clamp' }
	);

	const styleCombined: React.CSSProperties = {
		transform: `scale(${scale})`,
		opacity,
	};

	// Render based on the type of the current figure
	if (currentFigure.type === 'headline') {
		return (
			<div className="text-8xl font-semibold text-black text-center" style={styleCombined}>
				{currentFigure.content}
			</div>
		);
	} else if (currentFigure.type === 'figure') {
		return (
			<div className="flex w-full justify-center items-center">
				<Img className="object-fill min-h-[500px]" style={styleCombined} src={currentFigure.content} />
			</div>
		);
	} else if (currentFigure.type === 'equation') {
		return (
			<div className="text-8xl font-semibold text-black text-center" style={styleCombined}>
				<InlineMath math={currentFigure.content} />
			</div>
		);
	}

	return null;
};
