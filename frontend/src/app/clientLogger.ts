const clientLogger = {
    log: (message: string) => {
      console.log(`[ArxFlix] ${message}`);
    },
    error: (message: string) => {
      console.error(`[ArxFlix] ${message}`);
    }
  };

  export default clientLogger;
