import { NextResponse } from 'next/server';
import axios from 'axios';
import logger from '../logger';
import { GENERATE_ASSETS_ENDPOINT } from './endpointConfig';
import { BASE_URL, HEADERS } from '../baseConfig';

interface RequestBody {
  script: string;
  mp3_output: string;
  srt_output: string;
  rich_output: string;
}

interface ApiResponse {
  status: string;
  total_duration?: number;
  error?: string;
}

/**
 * Handles POST requests to generate assets from a script.
 *
 * @param {Request} request - The incoming request object.
 * @returns {Promise<NextResponse>} - The response object.
 */
export async function POST(request: Request): Promise<NextResponse> {
  try {
    const requestBody = await parseRequestBody(request);
    validateRequestBody(requestBody);

    const response = await sendAssetsToApi(requestBody);

    return handleApiResponse(response, requestBody);
  } catch (error) {
    logger.error(`Error in POST handler: ${error.message}`);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

async function parseRequestBody(request: Request): Promise<RequestBody> {
  try {
    return await request.json();
  } catch (error) {
    throw new Error('Invalid JSON in request body');
  }
}

function validateRequestBody(body: RequestBody): void {
  const { script, mp3_output, srt_output, rich_output } = body;
  if (!script || !mp3_output || !srt_output || !rich_output) {
    throw new Error('Script, mp3_output, srt_output, and rich_output parameters are required');
  }
}

async function sendAssetsToApi(body: RequestBody): Promise<ApiResponse> {
  try {
    const response = await axios.post<ApiResponse>(`${BASE_URL}${GENERATE_ASSETS_ENDPOINT}`, {
      script: body.script,
      mp3_output: body.mp3_output,
      srt_output: body.srt_output,
      rich_output: body.rich_output,
      use_path: false,
    }, { headers: HEADERS });

    return response.data;
  } catch (error) {
    throw new Error('Failed to fetch data from API');
  }
}

function handleApiResponse(response: ApiResponse, requestBody: RequestBody): NextResponse {
  if (response.status === 'ERR') {
    const error: string = response.error as string; // Explicitly type 'error' as a string
    logger.error(`API responded with an error: ${error}`);
    throw new Error('Error fetching the paper data.');
  }

  return NextResponse.json({
    mp3_output: requestBody.mp3_output,
    srt_output: requestBody.srt_output,
    rich_output: requestBody.rich_output,
    total_duration: response.total_duration,
  });
}
