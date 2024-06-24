import { NextResponse } from 'next/server';
import axios from 'axios';
import logger from '../logger';
import { BASE_URL, HEADERS } from '../baseConfig';
import { GENERATE_PAPER_ENDPOINT } from './endpointConfig';

interface ApiResponse {
    status: string;
    data?: any;
    error?: string;
  }

/**
 * Handles GET requests to generate a paper from a URL.
 *
 * @param {Request} request - The incoming request object.
 * @returns {Promise<NextResponse>} - The response object.
 */
export async function GET(request: Request): Promise<NextResponse> {
    try {
        const url = getUrlFromRequest(request);
        validateUrl(url);

        const response = await fetchPaperFromApi(url);

        return handleApiResponse(response);
    } catch (error) {
        logger.error(`Error in GET handler: ${error.message}`);
        return NextResponse.json({ error: error.message }, { status: 500 });
    }
}

function getUrlFromRequest(request: Request): string | null {
    const { searchParams } = new URL(request.url);
    return searchParams.get('url');
}

function validateUrl(url: string | null): void {
    if (!url) {
        throw new Error('URL parameter is required');
    }
}

async function fetchPaperFromApi(url: string): Promise<ApiResponse> {
    try {
        const response = await axios.get<ApiResponse>(`${BASE_URL}${GENERATE_PAPER_ENDPOINT}`, {
            params: { url },
            headers: HEADERS,
        });
        return response.data;
    } catch (error) {
        throw new Error('Failed to fetch data from API');
    }
}

function handleApiResponse(response: ApiResponse): NextResponse {
    if (response.status === 'ERR') {
        const error: string = response.error as string;
        logger.error(`API responded with an error: ${error}`);
        throw new Error('Error fetching the paper data.');
    }
    return NextResponse.json(response);
}
