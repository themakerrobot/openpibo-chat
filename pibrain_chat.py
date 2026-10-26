"""파이브레인에서: 파이보와 같은 대화 모델(themaker)·같은 도구 목록으로 이야기합니다.
다른 것은 기기 함수뿐입니다 (openpibo-os.pibrain 기준).

    sudo ~/.pyenv/bin/python3 pibrain_chat.py --url http://127.0.0.1:50020
    python3 pibrain_chat.py --fake       # openpibo가 없는 PC에서

| 도구 | 파이브레인 |
| --- | --- |
| 불빛 | DeviceByPiBrain.led_on(r, g, b) / led_off() (NeoPixel 1개, GPIO12, root 필요) |
| 보기 | Camera().read() + Detect().detect_object() (파이보와 같음) |
| 움직이기 | 서보가 없어 실행하지 않고 "움직일 수는 없어요"라고 말함 |

pibo_chat.py와 tools.py가 같은 폴더에 있어야 합니다. 마이크가 없어 글자로 입력합니다.
"""

from __future__ import annotations

import threading

import pibo_chat
from tools import COLORS


class PiBrain(pibo_chat.Pibo):
    name = "파이브레인"
    can_move = False

    def part(self, name: str):
        if name == "device" and name not in self._parts:
            from openpibo.device import DeviceByPiBrain
            self._parts[name] = DeviceByPiBrain()
        return super().part(name)

    def light(self, color: str) -> None:
        """LED: led_on(r, g, b) / led_off(). NeoPixel의 show()가 가끔 돌아오지 않아
        (openpibo-os.pibrain tools의 led_call과 같은 까닭) 2초 넘으면 실패로 봅니다."""
        rgb = COLORS[color]
        if self.fake:
            return
        dev = self.part("device")
        t = threading.Thread(target=dev.led_off if rgb is None else (lambda: dev.led_on(*rgb)),
                             daemon=True)
        t.start()
        t.join(2)
        if t.is_alive():
            raise TimeoutError("LED가 2초 안에 응답하지 않음")

    def move(self, action: str) -> None:
        raise RuntimeError("파이브레인은 몸이 없음")

    def stop(self) -> None:
        pass


if __name__ == "__main__":
    pibo_chat.main(PiBrain)
