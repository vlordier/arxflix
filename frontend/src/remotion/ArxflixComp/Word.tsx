import React from 'react';
import { interpolate, Easing } from 'remotion';
import { SubtitleItem } from 'parse-srt';
import { InlineMath } from 'react-katex';

// Main component to render words based on their type
export const Word: React.FC<{
	item: SubtitleItem;
	frame: number;
}> = ({ item, frame }) => {
	// Determine the type of word and render the appropriate component
	if (item.text.startsWith('*') && item.text.endsWith('*')) {
		return <BoldWord item={item} frame={frame} />;
	} else if (item.text.startsWith('$') && item.text.endsWith('$')) {
		return <LatexWord item={item} frame={frame} />;
	} else {
		return <ClassicalWord item={item} frame={frame} />;
	}
};

// Component to render a classical word
export const ClassicalWord: React.FC<{
	item: SubtitleItem;
	frame: number;
	stroke?: boolean;
}> = ({ item, frame, stroke = true }) => {
	// Calculate opacity and vertical shift for animation
	const opacity = interpolate(frame, [item.start, item.start + 15], [0, 1], {
		extrapolateLeft: 'clamp',
		extrapolateRight: 'clamp',
	});
	const shiftY = interpolate(frame, [item.start, item.start + 10], [0.25, 0], {
		easing: Easing.out(Easing.quad),
		extrapolateLeft: 'clamp',
		extrapolateRight: 'clamp',
	});

	return (
		<span
			style={{
				display: 'inline-block',
				opacity,
				transform: `translateY(${shiftY}em)`,
				paintOrder: 'stroke fill',
			}}
			className="text-black"
		>
			{item.text}
		</span>
	);
};

// Component to render a bold word
export const BoldWord: React.FC<{
	item: SubtitleItem;
	frame: number;
	stroke?: boolean;
}> = ({ item, frame, stroke = false }) => {
	// Calculate opacity and vertical shift for animation
	const opacity = interpolate(frame, [item.start, item.start + 15], [0, 1], {
		extrapolateLeft: 'clamp',
		extrapolateRight: 'clamp',
	});
	const shiftY = interpolate(frame, [item.start, item.start + 10], [0.25, 0], {
		easing: Easing.out(Easing.quad),
		extrapolateLeft: 'clamp',
		extrapolateRight: 'clamp',
	});

	return (
		<span
			style={{
				display: 'inline-block',
				opacity,
				transform: `translateY(${shiftY}em)`,
				color: 'blue',
				fontWeight: 'bold',
				WebkitTextStroke: stroke ? '0.15px black' : undefined,
				stroke: stroke ? '0.15px black' : undefined,
				paintOrder: 'stroke fill',
			}}
		>
			{item.text.slice(1, -1)}
		</span>
	);
};

// Component to render a LaTeX word
export const LatexWord: React.FC<{
	item: SubtitleItem;
	frame: number;
	stroke?: boolean;
}> = ({ item, frame, stroke = false }) => {
	// Calculate opacity and vertical shift for animation
	const opacity = interpolate(frame, [item.start, item.start + 15], [0, 1], {
		extrapolateLeft: 'clamp',
		extrapolateRight: 'clamp',
	});
	const shiftY = interpolate(frame, [item.start, item.start + 10], [0.25, 0], {
		easing: Easing.out(Easing.quad),
		extrapolateLeft: 'clamp',
		extrapolateRight: 'clamp',
	});

	return (
		<span
			style={{
				display: 'inline-block',
				opacity,
				transform: `translateY(${shiftY}em)`,
				WebkitTextStroke: stroke ? '0.15px black' : undefined,
				stroke: stroke ? '0.15px black' : undefined,
				paintOrder: 'stroke fill',
			}}
		>
			<InlineMath math={item.text.slice(1, -1)} />
		</span>
	);
};
