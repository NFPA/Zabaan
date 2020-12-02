"use strict";

/* eslint-env jquery, browser */
var count = -1;
var sentencePool = {
  current: undefined,
  content: undefined,
  pop: function pop() {
    this.current = this.content.pop()[0];
    return this.current;
  }
};
var translation;

function onEditButtonClick(idx) {
  $("#validation-display-".concat(idx)).addClass('hide');
  $("#validation-input-".concat(idx)).removeClass('hide');
  $("#validation-input-".concat(idx)).addClass('display-block');
  $("#validation-input-".concat(idx, " > input")).focus();
}

function onEditInputBlur(idx) {
  $("#validation-display-".concat(idx)).removeClass('hide');
  $("#validation-input-".concat(idx)).addClass('display-block');
  $("#validation-input-".concat(idx)).removeClass('display-block');
  $("#validation-input-".concat(idx)).addClass('hide');
  var v = $("#validation-input-".concat(idx, " > input")).val();
  $("#validation-display-".concat(idx)).text(v);
  onEditionFinish(idx, v);
}

function onEditionFinish(idx, value) {
  console.log('onEditionFinish', idx, value); // Use sentencePool.current for current displaying text
}

function onNextSentence(source) {
  $('.validation-row').remove();
  $.ajax({
    url: '/sentence',
    method: 'POST',
    data: {
      source: source
    },
    success: function success(res) {
      translation = res.content;
      var t = res.content.map(function (v, i) {
        return generateValidationRow(i, i, v);
      }).join('');
      $('#validate-segment').append(t);
      $('#validation-count').text("Completed: ".concat(++count));
    }
  });
}

function generateValidationRow(id, modelName, content) {
  return "\n<div id=\"validation-row-".concat(id, "\" class=\"ui secondary segment grid validation-row\">\n    <div class=\"three wide column centered row\">\n        <a class=\"ui basic label red\">").concat(modelName, "</a>\n    </div>\n    <div id=\"translation-column\" class=\"nine wide column\">\n        <p id=\"validation-display-").concat(id, "\">").concat(content, "</p>\n        <div id=\"validation-input-").concat(id, "\" class=\"ui fluid input hide\">\n            <input type=\"text\" value=\"").concat(content, "\" onblur=\"onEditInputBlur('").concat(id, "')\">\n        </div>\n    </div>\n    <div class=\"ui four wide column form\">\n        <div class=\"inline fields\">\n            <div class=\"field\">\n                <div class=\"ui radio checkbox red\">\n                    <input id=\"radio-checkbox-").concat(id, "-true\" type=\"radio\" name=\"").concat(id, "\" checked=\"checked\">\n                    <label></label>\n                </div>\n            </div>\n            <div class=\"field\">\n                <div class=\"ui radio checkbox\">\n                    <input type=\"radio\" name=\"").concat(id, "\">\n                    <label></label>\n                </div>\n            </div>\n            <div id=\"edit-button\" class=\"field\" onclick=\"onEditButtonClick('").concat(id, "')\">\n                <i class=\"edit icon red\"></i>\n            </div>\n        </div>\n    </div>\n</div>");
}

$('#submit-button').click(function (e) {
  // Submit
  $.ajax({
    url: '/feedback',
    method: 'post',
    data: {
      code: $('#code-dropdown').dropdown('get value'),
      raw: sentencePool.current,
      payload: JSON.stringify(translation.map(function (v, i) {
        return {
          translation: v,
          status: !!$("#validation-row-".concat(i, " #radio-checkbox-").concat(i, "-true")).prop('checked'),
          edit: $("#validation-input-".concat(i, " > input")).val()
        };
      }))
    },
    success: function success(data, textStatus, jqXHR) {
      console.log(data);
    }
  }); // Next

  $('#source-text').val(sentencePool.pop());
  onNextSentence(sentencePool.current);
});
$('#code-dropdown').dropdown('setting', 'onChange', function (code) {
  $.ajax({
    url: '/sentence',
    data: {
      code: code.toUpperCase()
    },
    success: function success(res) {
      $('#edit-header').removeClass('hide');
      sentencePool.content = res.content;
      $('#source-text').val(sentencePool.pop());
      onNextSentence(sentencePool.current);
    }
  });
});
$.ajax({
  url: '/codes',
  method: 'GET',
  success: function success(data) {
    var d = data.ok ? data.content : [];
    var res = d.map(function (v) {
      return "<div class=\"item\">".concat(v, "</div>");
    });
    $('#code-list').html(res);
  }
});