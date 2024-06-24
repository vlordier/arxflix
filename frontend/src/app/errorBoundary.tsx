'use client';

import React, { ReactNode, useState, ErrorInfo } from "react";

let logger: any;
if (typeof window === "undefined") {
  // Server-side
  logger = require('./logger').default;
} else {
  // Client-side
  logger = require('./clientLogger').default;
}

interface ErrorBoundaryProps {
  children: ReactNode;
}

const ErrorBoundary: React.FC<ErrorBoundaryProps> = ({ children }) => {
  const [hasError, setHasError] = useState(false);

  const handleCatch = (error: Error, errorInfo: ErrorInfo) => {
    logger.error('Error occurred:', error);
    logger.error('Error Info:', errorInfo);
    setHasError(true);
  };

  if (hasError) {
    return <h1>Something went wrong. Please try again later.</h1>;
  }

  return (
    <ErrorBoundaryWrapper onCatch={handleCatch}>
      {children}
    </ErrorBoundaryWrapper>
  );
};

interface ErrorBoundaryWrapperProps {
  onCatch: (error: Error, errorInfo: ErrorInfo) => void;
  children: ReactNode;
}

class ErrorBoundaryWrapper extends React.Component<ErrorBoundaryWrapperProps> {
  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.props.onCatch(error, errorInfo);
  }

  render() {
    return this.props.children;
  }
}

export default ErrorBoundary;
