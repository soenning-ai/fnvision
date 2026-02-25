# INDEX

Automatisch generierter Navigationsindex fuer aktive Python-Dateien (fnvision_dev).
Stand: 2026-02-25 17:58:10 UTC

## Dateien

- Indexierte Dateien: `11`

### `/fnvision/__init__.py`

- Zeilen: `14`
- Keine Klassen/Funktionen gefunden

### `/fnvision/config.py`

- Zeilen: `300`
- `class FoveaOutput`: `23`-`71`
- `def FoveaOutput.zone_shapes`: `64`-`71`
- `class FoveaConfig`: `79`-`300`
- `def FoveaConfig.__post_init__`: `198`-`199`
- `def FoveaConfig._validate`: `201`-`236`
- `def FoveaConfig.sigma_norm`: `243`-`245`
- `def FoveaConfig.to_yaml`: `251`-`268`
- `def FoveaConfig.from_yaml`: `271`-`300`

### `/fnvision/encoder.py`

- Zeilen: `413`
- `class FoveaEncoder`: `21`-`318`
- `def FoveaEncoder.__init__`: `54`-`60`
- `def FoveaEncoder.encode`: `66`-`167`
- `def FoveaEncoder._compute_f_positions`: `173`-`211`
- `def FoveaEncoder._attention_factors`: `213`-`242`
- `def FoveaEncoder._sample_zones`: `244`-`318`
- `def _safe_crop`: `324`-`364`
- `def _apply_res_factor`: `367`-`384`
- `def _resize_f32`: `387`-`403`
- `def _smoothstep`: `406`-`413`

### `/fnvision/gaze.py`

- Zeilen: `180`
- `def _clamp`: `14`-`15`
- `class GazeState`: `19`-`24`
- `class GazeController`: `27`-`180`
- `def GazeController.__init__`: `41`-`56`
- `def GazeController.state`: `59`-`60`
- `def GazeController.copy_state`: `62`-`68`
- `def GazeController.snapshot`: `70`-`72`
- `def GazeController.reset`: `74`-`90`
- `def GazeController.f1_pos_norm`: `93`-`98`
- `def GazeController.f2_pos_norm`: `101`-`106`
- `def GazeController.step`: `108`-`167`
- `def GazeController._apply_bounds`: `169`-`180`

### `/fnvision/tools/__init__.py`

- Zeilen: `2`
- Keine Klassen/Funktionen gefunden

### `/fnvision/tools/calibration.py`

- Zeilen: `468`
- `def _noop`: `29`-`30`
- `def _clamp01`: `33`-`34`
- `def _to_bgr_u8`: `37`-`42`
- `def _parse_region`: `45`-`54`
- `class FrameSource`: `57`-`135`
- `def FrameSource.__init__`: `58`-`71`
- `def FrameSource._init_source`: `73`-`106`
- `def FrameSource.read_rgb`: `108`-`131`
- `def FrameSource.close`: `133`-`135`
- `class UIState`: `139`-`143`
- `def _mouse_cb`: `146`-`160`
- `def _build_parser`: `163`-`174`
- `def _create_trackbars`: `177`-`200`
- `def _read_cfg_from_trackbars`: `203`-`242`
- `def _sep01_to_abs`: `245`-`247`
- `def _panel`: `250`-`263`
- `def run_calibration`: `266`-`458`
- `def main`: `461`-`464`

### `/fnvision/weight_field.py`

- Zeilen: `88`
- `def _as_point_xy_norm`: `16`-`21`
- `def compute_weight_map`: `24`-`88`

### `/tests/__init__.py`

- Zeilen: `0`
- Keine Klassen/Funktionen gefunden

### `/tests/test_calibration.py`

- Zeilen: `97`
- `def test_parse_region_valid`: `11`-`14`
- `def test_parse_region_invalid`: `18`-`20`
- `def test_screen_source_read_failure_returns_none`: `23`-`37`
- `def test_trackbar_thresholds_clamp_para_zero`: `40`-`64`
- `def test_trackbar_auto_mode_enabled`: `67`-`90`
- `def test_sep01_to_abs_clamped`: `93`-`97`

### `/tests/test_encoder.py`

- Zeilen: `528`
- `def _make_frame`: `42`-`51`
- `def default_encoder`: `55`-`56`
- `def default_frame`: `60`-`61`
- `def default_result`: `65`-`66`
- `class TestCompositingInvariant`: `73`-`105`
- `def TestCompositingInvariant.test_solid_frame_preserved_center_gaze`: `83`-`93`
- `def TestCompositingInvariant.test_solid_frame_preserved_various_gaze`: `96`-`105`
- `class TestOutputDtypes`: `112`-`123`
- `def TestOutputDtypes.test_fovea_dtype`: `113`-`114`
- `def TestOutputDtypes.test_parafovea_dtype`: `116`-`117`
- `def TestOutputDtypes.test_periphery_dtype`: `119`-`120`
- `def TestOutputDtypes.test_weight_map_dtype`: `122`-`123`
- `class TestOutputRanges`: `130`-`144`
- `def TestOutputRanges.test_range`: `132`-`135`
- `def TestOutputRanges.test_range_random_noise_frame`: `137`-`144`
- `class TestOutputShapes`: `151`-`172`
- `def TestOutputShapes.test_fovea_shape`: `152`-`153`
- `def TestOutputShapes.test_parafovea_shape`: `155`-`156`
- `def TestOutputShapes.test_periphery_shape`: `158`-`159`
- `def TestOutputShapes.test_weight_map_shape_matches_frame`: `161`-`164`
- `def TestOutputShapes.test_custom_resolution`: `166`-`172`
- `class TestDeterminism`: `179`-`195`
- `def TestDeterminism.test_same_input_same_output`: `180`-`186`
- `def TestDeterminism.test_encoder_is_stateless`: `188`-`195`
- `class TestConvergence`: `202`-`218`
- `def TestConvergence.test_no_nan_inf_at_full_convergence`: `203`-`208`
- `def TestConvergence.test_f_separation_norm_zero_at_convergence`: `210`-`212`
- `def TestConvergence.test_f1_f2_collocated_at_convergence`: `214`-`218`
- `class TestEdgeGaze`: `225`-`242`
- `def TestEdgeGaze.test_no_crash_at_corners`: `227`-`230`
- `def TestEdgeGaze.test_no_nan_at_corners`: `233`-`236`
- `def TestEdgeGaze.test_out_of_range_gaze_clamped`: `238`-`242`
- `class TestAttentionExtremes`: `249`-`270`
- `def TestAttentionExtremes.test_no_nan_at_zero_attention`: `250`-`253`
- `def TestAttentionExtremes.test_no_nan_at_full_attention`: `255`-`258`
- `def TestAttentionExtremes.test_attention_echoed_in_output`: `260`-`264`
- `def TestAttentionExtremes.test_out_of_range_attention_clamped`: `266`-`270`
- `class TestZoomEffect`: `277`-`298`
- `def TestZoomEffect.test_fovea_separation_norm_wide_vs_zoomed`: `278`-`283`
- `def TestZoomEffect.test_fovea_outputs_differ_at_different_zoom`: `285`-`291`
- `def TestZoomEffect.test_f_separation_norm_wide_equals_max_norm`: `293`-`298`
- `class TestAspectRatio`: `305`-`322`
- `def TestAspectRatio.test_non_square_shapes_correct`: `307`-`314`
- `def TestAspectRatio.test_non_square_no_nan`: `317`-`322`
- `class TestFoveaConfigValidation`: `329`-`356`
- `def TestFoveaConfigValidation.test_invalid_focal_radius_zero`: `330`-`332`
- `def TestFoveaConfigValidation.test_invalid_focal_radius_too_large`: `334`-`336`
- `def TestFoveaConfigValidation.test_invalid_separation_order`: `338`-`340`
- `def TestFoveaConfigValidation.test_invalid_threshold_order`: `342`-`344`
- `def TestFoveaConfigValidation.test_invalid_weight_gamma_zero`: `346`-`348`
- `def TestFoveaConfigValidation.test_invalid_resolution_tuple`: `350`-`352`
- `def TestFoveaConfigValidation.test_valid_config_no_error`: `354`-`356`
- `class TestConfigYamlRoundTrip`: `363`-`389`
- `def TestConfigYamlRoundTrip.test_default_config_round_trips`: `364`-`376`
- `def TestConfigYamlRoundTrip.test_custom_config_round_trips`: `378`-`389`
- `class TestEncodeInputGuards`: `396`-`410`
- `def TestEncodeInputGuards.test_wrong_dtype_raises_type_error`: `397`-`400`
- `def TestEncodeInputGuards.test_wrong_ndim_raises_value_error`: `402`-`405`
- `def TestEncodeInputGuards.test_wrong_channels_raises_value_error`: `407`-`410`
- `class TestFoveaOutputMetadata`: `417`-`439`
- `def TestFoveaOutputMetadata.test_attention_level_echoed`: `418`-`421`
- `def TestFoveaOutputMetadata.test_f_separation_norm_range`: `423`-`427`
- `def TestFoveaOutputMetadata.test_f_positions_within_unit_square`: `429`-`433`
- `def TestFoveaOutputMetadata.test_zone_shapes_helper`: `435`-`439`
- `class TestWeightMap`: `446`-`466`
- `def TestWeightMap.test_weight_map_peak_near_center_gaze`: `447`-`456`
- `def TestWeightMap.test_weight_map_max_is_one`: `458`-`461`
- `def TestWeightMap.test_weight_map_min_is_nonneg`: `463`-`466`
- `class TestComputeWeightMap`: `473`-`528`
- `def TestComputeWeightMap.test_symmetric_points_symmetric_map`: `474`-`485`
- `def TestComputeWeightMap.test_convergence_single_peak`: `487`-`499`
- `def TestComputeWeightMap.test_output_dtype_float32`: `501`-`503`
- `def TestComputeWeightMap.test_output_range`: `505`-`508`
- `def TestComputeWeightMap.test_aspect_ratio_non_square_shape`: `510`-`514`
- `def TestComputeWeightMap.test_no_nan_small_sigma`: `516`-`520`
- `def TestComputeWeightMap.test_gamma_shaping_monotone`: `522`-`528`

### `/tests/test_gaze.py`

- Zeilen: `241`
- `def _dist`: `11`-`12`
- `class TestGazeControllerInit`: `15`-`27`
- `def TestGazeControllerInit.test_defaults`: `16`-`22`
- `def TestGazeControllerInit.test_initial_gaze_is_clamped`: `24`-`27`
- `class TestStepAndBounds`: `30`-`73`
- `def TestStepAndBounds.test_tick_increments`: `31`-`36`
- `def TestStepAndBounds.test_negative_dt_raises`: `38`-`41`
- `def TestStepAndBounds.test_nonfinite_dt_raises`: `44`-`47`
- `def TestStepAndBounds.test_dt_zero_is_noop_except_tick`: `49`-`55`
- `def TestStepAndBounds.test_target_clamped_to_unit_square`: `57`-`64`
- `def TestStepAndBounds.test_f_positions_are_always_in_bounds`: `66`-`73`
- `class TestSaccadeAndSpring`: `76`-`125`
- `def TestSaccadeAndSpring.test_saccade_has_overshoot_cap`: `77`-`83`
- `def TestSaccadeAndSpring.test_saccade_does_not_overshoot_target`: `85`-`90`
- `def TestSaccadeAndSpring.test_pull_strength_zero_keeps_separation_constant`: `92`-`101`
- `def TestSaccadeAndSpring.test_pull_moves_separation_toward_max`: `103`-`113`
- `def TestSaccadeAndSpring.test_separation_is_clamped_for_large_dt`: `115`-`125`
- `class TestDeterminismPhaseA`: `128`-`138`
- `def TestDeterminismPhaseA.test_different_rngs_same_result_in_phase_a`: `129`-`138`
- `class TestStochasticPhaseB`: `141`-`241`
- `def TestStochasticPhaseB._target_at`: `142`-`143`
- `def TestStochasticPhaseB._state_tuple`: `145`-`151`
- `def TestStochasticPhaseB._pose_tuple`: `153`-`158`
- `def TestStochasticPhaseB.test_seed_reproducibility_over_100_ticks`: `160`-`168`
- `def TestStochasticPhaseB.test_hold_probability_effect`: `170`-`188`
- `def TestStochasticPhaseB.test_jitter_bounds_fuzz_1000_ticks`: `190`-`198`
- `def TestStochasticPhaseB.test_jitter_off_matches_phase_a_behavior`: `200`-`208`
- `def TestStochasticPhaseB.test_dt_zero_consumes_no_rng_and_no_jitter`: `210`-`219`
- `def TestStochasticPhaseB.test_replay_reproducibility_with_snapshot_reset`: `221`-`241`
