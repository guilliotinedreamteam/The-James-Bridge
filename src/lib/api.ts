export const api = {
  train: async () => {
    console.log("Training started");
    return Promise.resolve();
  },
  synthesize: async (phonemes: string) => {
    console.log("Synthesizing", phonemes);
    return Promise.resolve();
  }
};
