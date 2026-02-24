"""
Dynamic post-pipeline steps.

Runs after all M1 iterations complete: evaluation summary,
identification pipeline (M2), and optional cleanup.
"""
import traceback
from pathlib import Path


def _run_dynamic_post_pipeline(
    output_path: str,
    house_id: str,
    threshold_schedule: list,
    iterations_completed: int,
    all_device_profiles: dict,
    minimal_output: bool,
    skip_identification: bool,
    logger,
):
    """Run dynamic-specific post-pipeline steps: eval summary, identification, cleanup."""
    import time

    # 1. Generate threshold-independent evaluation summary (disaggregation metric)
    try:
        from disaggregation.rectangle.pipeline.evaluation_summary import generate_dynamic_evaluation_summary
        generate_dynamic_evaluation_summary(
            output_path=output_path,
            house_id=house_id,
            threshold_schedule=threshold_schedule,
            iterations_completed=iterations_completed,
            logger=logger,
        )
    except Exception as e:
        logger.error(f"Error generating dynamic evaluation summary: {e}")
        logger.error(traceback.format_exc())

    # 2. Run identification pipeline (session grouping + classification)
    if not skip_identification:
        try:
            from identification import (
                load_all_matches, filter_transient_events,
                classify_events,
                build_session_json,
            )
            logger.info(f"{'=' * 60}")
            logger.info("IDENTIFICATION PIPELINE (classify-first)")
            logger.info(f"{'=' * 60}")

            experiment_dir = Path(output_path)

            t0 = time.time()
            all_matches = load_all_matches(experiment_dir, house_id, threshold_schedule)
            logger.info(f"  Load matches: {len(all_matches)} matches ({time.time() - t0:.1f}s)")

            t0 = time.time()
            filtered, spike_stats = filter_transient_events(all_matches)
            spike_count = spike_stats.get('spike_count', 0) if isinstance(spike_stats, dict) else 0
            logger.info(f"  Filter spikes: {spike_count} removed, "
                        f"{len(filtered)} remaining ({time.time() - t0:.1f}s)")

            t0 = time.time()
            classified = classify_events(filtered)
            total_classified = sum(len(v) for v in classified.values())
            non_unknown = total_classified - len(classified.get('unknown', []))
            logger.info(f"  Classify: {non_unknown}/{total_classified} classified "
                        f"({time.time() - t0:.1f}s)")

            t0 = time.time()
            json_path = build_session_json(
                classified_sessions=classified,
                house_id=house_id,
                threshold_schedule=threshold_schedule,
                experiment_dir=experiment_dir,
                device_profiles=all_device_profiles,
                spike_stats=spike_stats,
            )
            logger.info(f"  Save JSON output ({time.time() - t0:.1f}s)")
            logger.info(f"  Device sessions JSON saved to {json_path}")
        except Exception as e:
            logger.error(f"Error in identification pipeline: {e}")
            logger.error(traceback.format_exc())

    # 3. Optional cleanup of intermediate files
    if minimal_output:
        try:
            from identification.cleanup import cleanup_intermediate_files
            cleanup_intermediate_files(Path(output_path), house_id, iterations_completed, logger)
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            logger.error(traceback.format_exc())
