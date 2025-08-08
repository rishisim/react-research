import requests
from bs4 import BeautifulSoup, Comment
from config import WEBSHOP_URL, ACTION_TO_TEMPLATE

def clean_str(p):
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")

def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return element.parent.name not in ignore and not isinstance(element, Comment)

def webshop_text(session, page_type, **kwargs):
    options = kwargs.get('options', {})
    try:
        url_map = {
            'init': f'{WEBSHOP_URL}/{session}',
            'search': f'{WEBSHOP_URL}/search_results/{session}/{kwargs.get("query_string", "")}/{kwargs.get("page_num", 1)}',
            'item': f'{WEBSHOP_URL}/item_page/{session}/{kwargs.get("asin", "")}/{kwargs.get("query_string", "")}/{kwargs.get("page_num", 1)}/{options}',
            'item_sub': f'{WEBSHOP_URL}/item_sub_page/{session}/{kwargs.get("asin", "")}/{kwargs.get("query_string", "")}/{kwargs.get("page_num", 1)}/{kwargs.get("subpage", "")}/{options}',
            'end': f'{WEBSHOP_URL}/done/{session}/{kwargs.get("asin", "")}/{options}'
        }
        url = url_map.get(page_type)
        if not url: raise ValueError(f"Invalid page_type: {page_type}")

        html = requests.get(url).text
        html_obj = BeautifulSoup(html, 'html.parser')
        texts = html_obj.find_all(string=True)
        visible_texts = list(filter(tag_visible, texts))

        # Match notebook formatting exactly
        observation = ''
        option_type = ''
        page_options = {}
        asins = []
        cnt = 0
        prod_cnt = 0
        just_prod = 0

        for t in visible_texts:
            if t == '\n': continue
            if t.replace('\n', '').replace('\\n', '').replace(' ', '') == '': continue

            if t.parent.name == 'button':  # button
                processed_t = f'\n[{t}] '
            elif t.parent.name == 'label':  # options
                if f"'{t}'" in url:
                    processed_t = f'[[{t}]]'
                else:
                    processed_t = f'[{t}]'
                page_options[str(t)] = option_type
            elif t.parent.get('class') == ["product-link"]: # product asins
                processed_t = f'\n[{t}] '
                if prod_cnt >= 3:
                    processed_t = ''
                prod_cnt += 1
                asins.append(str(t))
                just_prod = 0
            else: # regular, unclickable text
                processed_t = '\n' + str(t) + ' '
                if cnt < 2 and page_type != 'init': processed_t = ''
                if just_prod <= 2 and prod_cnt >= 4: processed_t = ''
                option_type = str(t)
                cnt += 1
            just_prod += 1
            observation += processed_t

        info = {'asins': asins, 'option_types': page_options}
        if 'Your score (min 0.0, max 1.0)' in visible_texts:
            idx = visible_texts.index('Your score (min 0.0, max 1.0)')
            info['reward'] = float(visible_texts[idx + 1])
            observation = 'Your score (min 0.0, max 1.0): ' + (visible_texts[idx + 1])
        return clean_str(observation), info
    except requests.exceptions.RequestException as e:
        return f"Error connecting to WebShop: {e}", {'error': str(e)}

class WebShopEnv:
    def __init__(self):
        self.sessions = {}

    def reset(self, session):
        """Reset the environment for a given session."""
        self.sessions[session] = {'session': session, 'page_type': 'init'}
        observation, info = webshop_text(session=session, page_type='init')
        return observation

    def step(self, session, action):
        done = False
        observation_ = None
        action_type = action.split('[')[0]

        if action_type == 'reset':
            self.sessions[session] = {'session': session, 'page_type': 'init'}  # Match notebook format
        elif action_type == 'think': pass
        elif action_type == 'search':
            assert self.sessions[session]['page_type'] == 'init'
            query = action[7:-1]
            self.sessions[session] = {'session': session, 'page_type': 'search', 'query_string': query, 'page_num': 1}
        elif action_type == 'click':
            button = action[6:-1]
            page_type = self.sessions[session]['page_type']
            if button == 'Buy Now':
                assert page_type == 'item'
                self.sessions[session]['page_type'] = 'end'
                done = True
            elif button == 'Back to Search':
                assert page_type in ['search', 'item_sub', 'item']
                self.sessions[session] = {'session': session, 'page_type': 'init'}
            elif button == '< Prev':
                assert page_type in ['search', 'item_sub', 'item']
                if page_type == 'item_sub':
                    self.sessions[session]['page_type'] = 'item'
                elif page_type == 'item':
                    self.sessions[session]['page_type'] = 'search'
                    self.sessions[session]['options'] = {}  # Clear options when going back
            elif button == 'Next >':
                assert page_type == 'search'
                self.sessions[session]['page_num'] += 1
            elif button in ACTION_TO_TEMPLATE:
                assert page_type == 'item'  # Only from main item page
                self.sessions[session]['page_type'] = 'item_sub'
                self.sessions[session]['subpage'] = button
            else:
                if page_type == 'search':
                    assert button in self.sessions[session].get('asins', [])  # must be asins
                    self.sessions[session]['page_type'] = 'item'
                    self.sessions[session]['asin'] = button
                elif page_type == 'item':
                    assert 'option_types' in self.sessions[session]
                    assert button in self.sessions[session]['option_types'], (button, self.sessions[session]['option_types'])  # must be options
                    option_type = self.sessions[session]['option_types'][button]
                    if 'options' not in self.sessions[session]:
                        self.sessions[session]['options'] = {}
                    self.sessions[session]['options'][option_type] = button
                    observation_ = f'You have clicked {button}.'
        else:
            assert False, f"Invalid action format: {action}"

        observation, info = webshop_text(session=session, **{k:v for k,v in self.sessions[session].items() if k != 'session'})
        if 'error' in info: return observation, 0.0, True
        if observation_: observation = observation_
        self.sessions[session].update(info)
        return observation, info.get('reward', 0.0), done
