export type ConfigKey = "XMglass" | "DJI" | "Multiview";

export interface Entry {
  config: ConfigKey;
  slice_id: string;
  video_name: string;
  video_path: string;
  media_kind: "video" | "image" | "other";
  video_exists: boolean;
  scene: string;
  operation: string;
  protocol_name: string;
  has_protocol: boolean;
  issue: string;
  length: string;
  time_stamp: string;
  tools: string;
  gpt4o_output: string;
  date: string;
}

export interface Metadata {
  entries: Entry[];
  protocols: Record<ConfigKey, Record<string, string>>;
}
