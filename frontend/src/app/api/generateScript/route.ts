import { NextResponse } from 'next/server';
import axios from 'axios';
import logger from '../logger';
import { BASE_URL, HEADERS } from '../baseConfig';
import { GENERATE_SCRIPT_ENDPOINT } from './endpointConfig';

interface RequestBody {
  paper: string;
}

interface ApiResponse {
  status: string;
  data?: any;
  error?: string;
}

/**
 * Handles POST requests to generate a script from a paper.
 *
 * @param {Request} request - The incoming request object.
 * @returns {Promise<NextResponse>} - The response object.
 */
export async function POST(request: Request): Promise<NextResponse> {
  try {
    const requestBody = await parseRequestBody(request);
    validatePaper(requestBody.paper);

    const response = await sendPaperToApi(requestBody.paper);

    return handleApiResponse(response);
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

function validatePaper(paper: string | undefined): void {
  if (!paper) {
    throw new Error('Paper parameter is required');
  }
}

async function sendPaperToApi(paper: string): Promise<ApiResponse> {
  try {
    const response = await axios.post<ApiResponse>(`${BASE_URL}${GENERATE_SCRIPT_ENDPOINT}`, { paper, use_path: false }, { headers: HEADERS });
    return response.data;
  } catch (error) {
    throw new Error('Failed to fetch data from API');
  }
}

function handleApiResponse(response: ApiResponse): NextResponse {
  if (response.status === 'ERR') {
    logger.error(`API responded with an error: ${response.error}`);
    throw new Error('Error fetching the paper data.');
  }

  return NextResponse.json(response);
}
