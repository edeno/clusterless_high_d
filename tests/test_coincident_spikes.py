"""Unit tests for cross-probe coincident-spike artifact detection.

Ported from continuum_swr_replay's test_spikes.py. Pure-numpy tests; no
Spyglass/DataJoint connection required.
"""

from __future__ import annotations

import numpy as np

from clusterless_high_d.data_loaders.spikes import detect_coincident_spikes


class TestCoincidentSpikeDetection:
    """Test artifact removal via coincident spike detection."""

    def test_empty_spike_times_returns_empty_lists(self):
        spike_times = [np.array([]) for _ in range(4)]
        filtered_times, filtered_indices = detect_coincident_spikes(spike_times)
        assert len(filtered_times) == 0
        assert len(filtered_indices) == 0

    def test_single_group_passes_through(self):
        spike_times = [np.array([1.0, 2.0, 3.0])]
        filtered_times, _ = detect_coincident_spikes(spike_times)
        assert len(filtered_times) == 1
        np.testing.assert_array_equal(filtered_times[0], spike_times[0])

    def test_non_coincident_spikes_all_preserved(self):
        spike_times = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.1, 2.1, 3.1]),  # 100ms apart (>> 40us threshold)
            np.array([1.2, 2.2, 3.2]),
            np.array([1.3, 2.3, 3.3]),
        ]
        filtered_times, _ = detect_coincident_spikes(spike_times)
        total_original = sum(len(st) for st in spike_times)
        total_filtered = sum(len(st) for st in filtered_times)
        assert total_filtered == total_original

    def test_coincident_spikes_removed_correctly(self):
        spike_times = [
            np.array([1.0, 2.0, 5.0]),
            np.array([1.00002, 3.0, 5.00001]),  # 20us coincidences
            np.array([1.00003, 4.0, 5.00002]),  # 30us coincidences
            np.array([6.0, 7.0, 8.0]),
        ]
        filtered_times, _ = detect_coincident_spikes(spike_times)
        total_original = sum(len(st) for st in spike_times)
        total_filtered = sum(len(st) for st in filtered_times)
        assert total_filtered < total_original
        assert total_original - total_filtered >= 2

        all_filtered = np.concatenate(filtered_times)
        for clean in (2.0, 3.0, 4.0, 6.0):
            assert np.any(np.abs(all_filtered - clean) < 0.01)

    def test_two_of_four_coincident_above_threshold_removed(self):
        """2/4 = 0.5 > 0.33: both members of the pair must be removed."""
        spike_times = [
            np.array([1.0, 5.0]),
            np.array([1.00002, 6.0]),  # 20us from the leading spike
            np.array([3.0, 7.0]),
            np.array([4.0, 8.0]),
        ]
        filtered_times, _ = detect_coincident_spikes(spike_times)
        all_filtered = np.concatenate([ft for ft in filtered_times if len(ft)])
        assert not np.any(np.abs(all_filtered - 1.0) < 0.001)
        assert not np.any(np.abs(all_filtered - 1.00002) < 0.001)
        for clean in (3.0, 4.0, 5.0, 6.0, 7.0, 8.0):
            assert np.any(np.abs(all_filtered - clean) < 0.001)

    def test_well_separated_spikes_preserved(self):
        spike_times = [
            np.array([1.0, 5.0, 9.0]),
            np.array([2.0, 6.0, 10.0]),
            np.array([3.0, 7.0, 11.0]),
        ]
        filtered_times, _ = detect_coincident_spikes(spike_times)
        assert sum(len(st) for st in filtered_times) == sum(
            len(st) for st in spike_times
        )

    def test_output_length_matches_input_length(self):
        spike_times = [
            np.array([1.0, 2.0, 5.0]),
            np.array([1.00001, 5.00001]),
            np.array([1.00002, 3.0, 5.00002]),
            np.array([6.0, 7.0, 8.0]),
        ]
        filtered_times, filtered_indices = detect_coincident_spikes(spike_times)
        assert len(filtered_times) == len(spike_times)
        assert len(filtered_indices) == len(spike_times)

    def test_empty_groups_get_empty_arrays(self):
        spike_times = [
            np.array([1.00000, 3.0, 4.0]),
            np.array([1.00001]),  # only coincident -> should become empty
            np.array([1.00002, 5.0, 6.0]),
            np.array([1.00003, 7.0, 8.0]),
        ]
        filtered_times, filtered_indices = detect_coincident_spikes(spike_times)
        assert len(filtered_times) == 4
        assert len(filtered_indices) == 4
        assert len(filtered_times[1]) == 0
        assert len(filtered_indices[1]) == 0
        assert filtered_times[1].dtype == np.float64
        assert filtered_indices[1].dtype == np.intp
        assert 1.00000 not in filtered_times[0]
        assert 3.0 in filtered_times[0]
        assert 4.0 in filtered_times[0]

    def test_waveform_feature_alignment_after_filtering(self):
        spike_times = [
            np.array([1.0, 2.0, 5.0]),
            np.array([1.00001, 5.00001]),
            np.array([1.00002, 3.0, 5.00002]),
            np.array([6.0, 7.0, 8.0]),
        ]
        waveform_features = [
            np.arange(len(st) * 4).reshape(-1, 4).astype(float) + i * 100
            for i, st in enumerate(spike_times)
        ]
        filtered_times, filtered_indices = detect_coincident_spikes(spike_times)
        filtered_features = [
            features[ind]
            for ind, features in zip(filtered_indices, waveform_features, strict=True)
        ]
        for i in range(len(spike_times)):
            assert len(filtered_features[i]) == len(filtered_times[i])
        assert len(filtered_features[3]) == 3
        assert filtered_features[3][0, 0] >= 300

    def test_first_spike_in_artifact_cluster_also_removed(self):
        spike_times = [
            np.array([1.00000, 5.0]),
            np.array([1.00001, 6.0]),
            np.array([1.00002, 7.0]),
            np.array([1.00003, 8.0]),
        ]
        filtered_times, _ = detect_coincident_spikes(spike_times)
        all_filtered = np.concatenate(filtered_times)
        survivors = all_filtered[(all_filtered >= 0.999) & (all_filtered <= 1.001)]
        assert len(survivors) == 0
        for clean in (5.0, 6.0, 7.0, 8.0):
            assert np.any(np.abs(all_filtered - clean) < 0.01)

    def test_preserves_spike_order(self):
        spike_times = [
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([1.5, 2.5, 3.5, 4.5]),
            np.array([1.7, 2.7, 3.7, 4.7]),
        ]
        filtered_times, _ = detect_coincident_spikes(spike_times)
        for filtered in filtered_times:
            if len(filtered) > 1:
                assert np.all(filtered[:-1] <= filtered[1:])

    def test_threshold_tuning_changes_removal(self):
        """A looser fraction threshold removes a 1-of-3 (0.33) coincidence too."""
        spike_times = [
            np.array([1.0, 5.0]),
            np.array([1.00002, 6.0]),  # 2-of-3 = 0.67 coincidence at t~1
            np.array([3.0, 7.0]),
        ]
        # default 0.33: 0.67 > 0.33 -> removed
        kept_default, _ = detect_coincident_spikes(spike_times)
        all_default = np.concatenate([k for k in kept_default if len(k)])
        assert not np.any(np.abs(all_default - 1.0) < 1e-3)
        # strict 0.9: 0.67 < 0.9 -> kept
        kept_strict, _ = detect_coincident_spikes(
            spike_times, max_coincident_fraction=0.9
        )
        all_strict = np.concatenate([k for k in kept_strict if len(k)])
        assert np.any(np.abs(all_strict - 1.0) < 1e-3)
