# def _get_audio_meta(file_path: str, minimal: bool = True) -> AudioMeta:
#     """AudioMeta from a path to an audio file.

#     Args:
#         file_path (str): Resolved path of valid audio file.
#         minimal (bool): Whether to only load the minimal set of metadata (takes longer if not).
#     Returns:
#         AudioMeta: Audio file path and its metadata.
#     """
#     info = audio_info(file_path)
#     amplitude: tp.Optional[float] = None
#     if not minimal:
#         wav, sr = audio_read(file_path)
#         amplitude = wav.abs().max().item()
#     return AudioMeta(file_path, info.duration, info.sample_rate, amplitude)
