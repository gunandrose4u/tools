import os
import sys
import pathlib

curdir = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(curdir))


SINGLESTREAM = "single_stream"
OFFLINE = "offline"
MULTISTREAM = "multi_stream"
SERVER = "server"

PERFORMANCEONLY_MODE = "PerformanceOnly"
ACCURACYONLY_MODE = "AccuracyOnly"

DEFAULT_MLPERF_CONFIG = os.path.join(curdir, "mlperf.conf")


# MLPerf default settings, for values allowed in test settings, see
# https://github.com/mlcommons/inference/blob/master/loadgen/test_settings.h
MLPERF_DEDAULT_SETTINGS = {
    "userConf": "",
    "logSettings": {
        "outputDir": "mlperf_results",
        "copySummaryToStdout": True,
        "enableTrace": False
    },
    "testSettings": {
        "fromFile": "",
        "overrideValues": {
            "single_stream_expected_latency_ns": 1131321,
            "offline_expected_qps": 10
        }
    }
}