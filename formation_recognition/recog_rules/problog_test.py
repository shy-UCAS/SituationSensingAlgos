import time

from problog.program import PrologFile, PrologString
from problog import get_evaluatable

def rules_multi_replace(text, replace_dict):
    for old, new in replace_dict.items():
        text = text.replace(old, new)
    
    return text

if __name__ == "__main__":
    with open('census_infer.pl', 'rt') as rfid:
        problog_model_str = rfid.read()

    replace_vars = {'VarSpeedChangeEps': '0.5', # 判定无人机加速的最小速度提升比例
                    'VarInterAngleEps': '20', # 判定无人机角度改变的最小偏向角度
                    'VarDistChangeEps': '0.1', # 判定无人机距离变化的最小距离缩小值
                    }
    
    problog_model_str = rules_multi_replace(problog_model_str, replace_vars)    
    # print("rules: %s" % problog_model_str)

    test_sw = 1
    if test_sw == 1:
        new_knows = """        
        0.8::avg_dist(euav1, hq1, 3.4, '2022-09-10 07:32:11.321').
        0.9::avg_dist(euav1, hq1, 2.1, '2022-09-10 07:32:12.112').
        
        0.89::avg_speed(euav1, 5, '2023-02-13 05:23:11.39').
        0.78::avg_speed(euav1, 8, '2023-02-13 05:23:11.97').
        0.76::avg_speed(euav1, 3, '2023-02-13 05:23:13.12').
        
        0.86::avg_orient(euav2, 172, '2021-03-21 11:03:21.371').
        0.86::avg_orient(euav2, -165, '2021-03-21 11:03:23.654').
        0.86::avg_orient(euav2, -114, '2021-03-21 11:03:25.765').
        """

        new_queries = """
        %query(targeting_facility(_, _)).
        %query(approach_facility(_, _)).
        query(speed_up(_, _)).
        query(slow_down(_, _)).
        query(orient_change(_, _, _)).
        query(orient_nochange(_, _, _)).
        query(avg_orient(_, _, _)).
        """

        # import pdb; pdb.set_trace()
        problog_program = PrologString(problog_model_str + '\n' + new_knows + '\n' + new_queries)

        _tic = time.time()
        # _cur_infer_program = PrologString(str(problog_model_compiled) + '\n' + new_knows + '\n' + new_queries)
        _result = get_evaluatable().create_from(problog_program).evaluate()

        for query, probability in _result.items():
            if probability > 0.0:
                print(query, probability)

        print("query evaluated in %.3fsecs" % (time.time() - _tic))