from pm1.plan import dates_in, plan


def test_fact_default():
    p = plan('What version of Milvus did I use?')
    assert p.intent == 'fact' and p.window is None and not p.subs


def test_timeline_with_iso_window_and_count():
    p = plan('How did my discussions about the RAG system progress from 2024-08-01 to 2024-10-22 in order? '
             'Mention ONLY and ONLY twenty items.')
    assert p.intent == 'timeline' and p.window == ('2024-08-01', '2024-10-22') and p.want == 20 and p.openers


def test_timeline_month_names_and_aspects():
    p = plan('Can you walk me through the order in which I brought up different aspects of my beach wedding '
             'planning from July 1, 2023 to July 6, 2023, including venue options, guest capacities, and permit fees?')
    assert p.intent == 'timeline' and p.window == ('2023-07-01', '2023-07-06')
    assert p.subs == ['venue options', 'guest capacities', 'permit fees']


def test_summary_request_vs_noun():
    assert plan('Can you summarize how my plans evolved from 2024-07-01 to 2024-07-29?').intent == 'timeline'
    assert plan('What strategies balance new material and revisiting summaries of chapters?').intent == 'fact'
    assert plan('Can you help me prove that the sequence x_n = 1/n approaches zero?').intent == 'fact'


def test_multi_between_events():
    p = plan('How many days are there between when I launch the testing suite development and when I start '
             'the deployment preparation?')
    assert p.intent == 'multi' and p.openers
    assert p.subs == ['I launch the testing suite development', 'I start the deployment preparation']


def test_multi_shared_head():
    p = plan('How much total delay have I noted across the agent updates, pedestrian updates, and camera data '
             'sync issues?')
    assert p.intent == 'multi'
    assert p.subs == ['delay noted the agent updates', 'delay noted pedestrian updates',
                      'delay noted camera data sync issues']


def test_total_as_noun_is_fact():
    assert plan('What detection rate and total number of test records did I mention?').intent == 'fact'


def test_chinese():
    p = plan('我从2024年8月1日到2024年10月22日的项目进展是怎样的？')
    assert p.intent == 'timeline' and p.window == ('2024-08-01', '2024-10-22')
    assert plan('我开始写测试和开始部署之间相隔多少天？').intent == 'multi'


def test_dates_in_month_only():
    assert dates_in('in March 2025')[0][1:] == ('2025-03-01', '2025-03-31')


def test_aggregate_flag():
    from pm1.plan import plan
    assert plan('How many projects have I led or am currently leading?').aggregate
    assert plan('What is the total number of online courses I have completed?').aggregate
    assert not plan('What is the name of my dentist?').aggregate
    assert plan('我一共去过几次健身房？').aggregate


def test_anchor_relative_dates():
    from pm1.plan import anchor, plan
    now = '2023/06/22 (Thu) 18:33'
    assert anchor('what did I do a week ago?', now) == ('2023-06-13', '2023-06-17')
    assert anchor('what did I eat yesterday', now) == ('2023-06-21', '2023-06-21')
    assert anchor("which airline did I fly on Valentine's day?", '2023/03/02') == ('2023-02-13', '2023-02-15')
    assert anchor('what did I do on christmas', now) == ('2022-12-24', '2022-12-26')   # latest one before now
    assert anchor('what did I buy last weekend', now) == ('2023-06-16', '2023-06-19')
    assert anchor('上个月我去了哪里', now) == ('2023-05-01', '2023-05-31')
    assert anchor('How many days ago did I meet Emma?', now) is None                  # asks for the distance
    assert anchor('what happened on 2023-03-01', now) is None                         # explicit date: _window
    assert plan('what did I do a week ago?').anchor is None                           # no `now`, no anchor
