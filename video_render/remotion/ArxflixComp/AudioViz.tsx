import { useAudioData, visualizeAudio } from '@remotion/media-utils';
import { linearPath } from "./waveform-path/waveform-path";
import React from 'react';
import {
	useCurrentFrame,
	useVideoConfig,
} from 'remotion';
import { extendViewBox } from "@remotion/paths";

export const fps = 30;

export const AudioViz: React.FC<{
	waveColor: string;
	numberOfSamples: number;
	freqRangeStartIndex: number;
	waveLinesToDisplay: number;
	mirrorWave: boolean;
	audioSrc: string;
	vizType?: 'bars' | 'waveform' | 'circle' | 'radialBars' | 'customWaveform';
}> = ({
	waveColor,
	numberOfSamples,
	freqRangeStartIndex,
	waveLinesToDisplay,
	mirrorWave,
	audioSrc,
	vizType = 'customWaveform',
}) => {
	const frame = useCurrentFrame();
	const { fps } = useVideoConfig();

	// Fetch audio data for the given audio source
	const audioData = useAudioData(audioSrc);

	if (!audioData) {
		return null; // Return null if audio data is not available
	}

	// Get frequency data for the current frame
	const frequencyData = visualizeAudio({
		fps,
		frame,
		audioData,
		numberOfSamples,
	});

	// Select a subset of frequency data
	const frequencyDataSubset = frequencyData.slice(
		freqRangeStartIndex,
		freqRangeStartIndex +
			(mirrorWave ? Math.round(waveLinesToDisplay / 2) : waveLinesToDisplay),
	);

	// Create mirrored frequency data if mirrorWave is true
	const frequenciesToDisplay = mirrorWave
		? [...frequencyDataSubset.slice(1).reverse(), ...frequencyDataSubset]
		: frequencyDataSubset;

	// Create a path for the custom waveform visualization
	const pathLogo = linearPath(frequenciesToDisplay, {
		type: 'steps',
		paths: [{ d: 'V', sy: 0, x: 50, ey: 100 }],
		height: 75,
		normalizeFactor: 40,
	});

	// Bar visualization
	const barViz = (
		<div className="flex flex-row h-48 items-center justify-center gap-2 mt-12">
			{frequenciesToDisplay.map((v, i) => (
				<div
					key={i}
					className="w-3 border rounded-lg"
					style={{
						minWidth: '1px',
						backgroundColor: waveColor,
						height: `${500 * Math.sqrt(v)}%`,
					}}
				/>
			))}
		</div>
	);

	// Waveform visualization
	const waveformViz = (
		<svg className="css-audio-viz" viewBox={extendViewBox("0 0 500 100", 1)} preserveAspectRatio="none">
			<polyline
				fill="none"
				stroke={waveColor}
				strokeWidth="2"
				points={frequenciesToDisplay
					.map((v, i) => `${(i / frequenciesToDisplay.length) * 500}, ${100 - (100 * Math.sqrt(v)) * 2}`)
					.join(' ')}
			/>
		</svg>
	);

	// Circle visualization
	const circleViz = (
		<div className="css-audio-viz" style={{ position: 'relative', width: '500px', height: '500px' }}>
			{frequenciesToDisplay.map((v, i) => {
				const angle = (i / frequenciesToDisplay.length) * 2 * Math.PI;
				const x = 250 + 200 * Math.cos(angle);
				const y = 250 + 200 * Math.sin(angle);
				return (
					<div
						key={i}
						className="css-bar"
						style={{
							position: 'absolute',
							left: `${x}px`,
							top: `${y}px`,
							width: '2px',
							height: `${100 * Math.sqrt(v)}px`,
							backgroundColor: waveColor,
							transform: `rotate(${angle}rad)`,
						}}
					/>
				);
			})}
		</div>
	);

	// Radial bars visualization
	const radialBarsViz = (
		<svg className="css-audio-viz" viewBox="0 0 500 500">
			{frequenciesToDisplay.map((v, i) => {
				const angle = (i / frequenciesToDisplay.length) * 2 * Math.PI;
				const x1 = 250 + 100 * Math.cos(angle);
				const y1 = 250 + 100 * Math.sin(angle);
				const x2 = 250 + (100 + 100 * Math.sqrt(v)) * Math.cos(angle);
				const y2 = 250 + (100 + 100 * Math.sqrt(v)) * Math.sin(angle);
				return (
					<line
						key={i}
						x1={x1}
						y1={y1}
						x2={x2}
						y2={y2}
						stroke={waveColor}
						strokeWidth="2"
					/>
				);
			})}
		</svg>
	);

	// Custom waveform visualization
	const customWaveform = (
		<svg className="css-audio-viz" viewBox={extendViewBox("0 0 700 75", 1)} preserveAspectRatio="none">
			<defs>
				<linearGradient id="lgrad" x1="0%" y1="50%" x2="100%" y2="50%">
					<stop offset="0%" style={{ stopColor: '#ff8d33', stopOpacity: 0.4 }} />
					<stop offset="50%" style={{ stopColor: '#ff8d33', stopOpacity: 1 }} />
					<stop offset="100%" style={{ stopColor: '#ff8d33', stopOpacity: 0.4 }} />
				</linearGradient>
			</defs>
			<path
				d={pathLogo}
				fill="none"
				stroke="url(#lgrad)"
				strokeWidth="5px"
				strokeLinecap="round"
			/>
		</svg>
	);

	// Return the appropriate visualization based on the vizType prop
	return vizType === 'bars' ? barViz :
		vizType === 'waveform' ? waveformViz :
		vizType === 'circle' ? circleViz :
		vizType === 'radialBars' ? radialBarsViz :
		vizType === 'customWaveform' ? customWaveform : null;
};
