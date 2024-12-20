import time

from problog.program import PrologFile, PrologString
from problog import get_evaluatable

if __name__ == "__main__":
    with open('census_infer.pl', 'rt') as rfid:
        problog_model_str = rfid.read()

    problog_model_str = problog_model_str.replace('VarInterAngleEps', '20')
    problog_model_str = problog_model_str.replace('VarDistChangeEps', '0.1')
    # print("rules: %s" % problog_model_str)

    test_sw = 1
    if test_sw == 1:
        new_knows = """        
        0.8::avg_dist(euav1, hq1, 3.4, '2022-09-10 07:32:11.321').
        0.9::avg_dist(euav1, hq1, 2.1, '2022-09-10 07:32:12.112').
        """

        new_queries = """
        :- use_module('census_infer.py').
        
        %query(targeting_facility(_, _)).
        query(approach_facility(_, _)).
        """

        problog_program = PrologString(problog_model_str + '\n' + new_knows + '\n' + new_queries)

        _tic = time.time()
        _result = get_evaluatable().create_from(problog_program).evaluate()

        for query, probability in _result.items():
            print(query, probability)

        print("query evaluated in %.3fsecs" % (time.time() - _tic))