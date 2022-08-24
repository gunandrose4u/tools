import threading
import time

from absl import app
import mlperf_loadgen


def load_samples_to_ram(query_samples):
    del query_samples
    return


def unload_samples_from_ram(query_samples):
    del query_samples
    return


# Processes queries in 3 slices that complete at different times.
def process_query_async(query_samples, i_slice):
    time.sleep(3 * (i_slice + 1))
    responses = []
    samples_to_complete = query_samples[i_slice:len(query_samples):3]
    for s in samples_to_complete:
        print(f"{s.id}-{s.index}")
        responses.append(mlperf_loadgen.QuerySampleResponse(s.id, 0, 0))
    mlperf_loadgen.QuerySamplesComplete(responses)


def issue_query(query_samples):
    threading.Thread(target=process_query_async,
                     args=(query_samples, 0)).start()
    #threading.Thread(target=process_query_async,
    #                 args=(query_samples, 1)).start()
    #threading.Thread(target=process_query_async,
    #                 args=(query_samples, 2)).start()


def flush_queries():
    pass


def process_latencies(latencies_ns):
    pass

def main(argv):
    del argv
    settings = mlperf_loadgen.TestSettings()
    settings.scenario = mlperf_loadgen.TestScenario.Offline
    settings.mode = mlperf_loadgen.TestMode.PerformanceOnly
    settings.offline_expected_qps = 1000

    sut = mlperf_loadgen.ConstructSUT(issue_query, flush_queries, process_latencies)
    qsl = mlperf_loadgen.ConstructQSL(
        1024, 128, load_samples_to_ram, unload_samples_from_ram)
    mlperf_loadgen.StartTest(sut, qsl, settings)
    mlperf_loadgen.DestroyQSL(qsl)
    mlperf_loadgen.DestroySUT(sut)


if __name__ == "__main__":
    app.run(main)