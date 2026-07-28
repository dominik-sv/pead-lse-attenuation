# from pathlib import Path
# import re
# import subprocess
# import sys
# import time


# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# SCRIPTS_DIR = PROJECT_ROOT / "scripts"
# REBUILD_SAMPLE_SIZE_MODULE = "src.tooling.aggregate_sample_size_all_years"
# MAX_ATTEMPTS = 4
# RETRY_DELAY_SECONDS = 20
# PIPELINE_SCRIPT_PATTERN = re.compile(r"^(?P<order>\d{2})(?P<suffix>[a-z]?)_.*\.py$")
# SCRIPT_ORDER_OVERRIDES = {
#     "00x_build_gbp_stock_universes.py": (0, 0),
#     "00b_collect_unique_gvkeys.py": (0, 1),
# }


# def numbered_pipeline_scripts() -> list[Path]:
#     numbered_scripts: list[tuple[int, int, str, str, Path]] = []
#     for script in SCRIPTS_DIR.glob("*.py"):
#         match = PIPELINE_SCRIPT_PATTERN.match(script.name)
#         if match is None or script.name == Path(__file__).name:
#             continue

#         order = int(match.group("order"))
#         if order >= 90:
#             continue

#         override_rank = SCRIPT_ORDER_OVERRIDES.get(script.name)
#         if override_rank is None:
#             stage_rank = (order, 999)
#         else:
#             stage_rank = override_rank

#         numbered_scripts.append(
#             (stage_rank[0], stage_rank[1], match.group("suffix"), script.name, script)
#         )

#     numbered_scripts.sort()
#     return [script for _, _, _, _, script in numbered_scripts]


# def run_with_retries(script_path: Path) -> None:
#     for attempt in range(1, MAX_ATTEMPTS + 1):
#         print(f"\nRunning {script_path.name} (attempt {attempt}/{MAX_ATTEMPTS})...")
#         result = subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT)
#         if result.returncode == 0:
#             print(f"{script_path.name} finished successfully.")
#             return

#         if attempt == MAX_ATTEMPTS:
#             raise SystemExit(
#                 f"{script_path.name} failed after {MAX_ATTEMPTS} attempts."
#             )

#         print(
#             f"{script_path.name} failed with exit code {result.returncode}. "
#             f"Retrying in {RETRY_DELAY_SECONDS} seconds..."
#         )
#         time.sleep(RETRY_DELAY_SECONDS)


# def rebuild_sample_size_all_years() -> None:
#     print("\nRebuilding sample_size_all_years.json from yearly sample_size.json files...")
#     result = subprocess.run(
#         [sys.executable, "-m", REBUILD_SAMPLE_SIZE_MODULE],
#         cwd=PROJECT_ROOT,
#     )
#     if result.returncode != 0:
#         raise SystemExit("Failed to rebuild sample_size_all_years.json.")


# def main() -> None:
#     for script_path in numbered_pipeline_scripts():
#         run_with_retries(script_path)
#     rebuild_sample_size_all_years()


# if __name__ == "__main__":
#     main()
