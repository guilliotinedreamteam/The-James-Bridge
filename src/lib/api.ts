export const api = {
  train: async () => {
    console.log("Training started (mock)");
    return Promise.resolve();
  },
  synthesize: async (phonemes: string) => {
    console.log(`Synthesizing: ${phonemes} (mock)`);
    return Promise.resolve();
  }
};
