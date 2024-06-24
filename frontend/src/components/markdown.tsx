import "@uiw/react-md-editor/markdown-editor.css";
import "@uiw/react-markdown-preview/markdown.css";
import dynamic from "next/dynamic";
import { Skeleton } from "@/components/ui/skeleton";
import type { FC } from 'react';

/**
 * Dynamic import for MDEditor component
 * SSR is disabled and a Skeleton loader is shown while loading
 */
export const MDEditor: FC = dynamic(() => import("@uiw/react-md-editor"), {
  ssr: false,
  loading: (): JSX.Element => (
    <div className="flex w-full">
      <Skeleton className="h-[700px] w-full" />
    </div>
  ),
});
