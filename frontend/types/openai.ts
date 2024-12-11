export interface OpenAIModel {
  id: string;
  name: string;
  maxLength: number; // maximum length of a message
  tokenLimit: number;
}

export enum OpenAIModelID {
  CUSTOM_1 = 'simple-conversation-chat',
  CUSTOM_2 = 'summary-conversation-chat'
}

// in case the `DEFAULT_MODEL` environment variable is not set or set to an unsupported model
export const fallbackModelID = OpenAIModelID.CUSTOM_1;

export const OpenAIModels: Record<OpenAIModelID, OpenAIModel> = {
  [OpenAIModelID.CUSTOM_1]: {
    id: OpenAIModelID.CUSTOM_1,
    name: 'SIMPLE',
    maxLength: 12000,
    tokenLimit: 3000,
  },
  [OpenAIModelID.CUSTOM_2]: {
    id: OpenAIModelID.CUSTOM_2,
    name: 'SUMMARY',
    maxLength: 12000,
    tokenLimit: 3000,
  },
};
