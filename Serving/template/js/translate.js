"use strict";

/* eslint-env jquery, browser */
var tgtLang = 'es';
$('#src-text').on('change', function (e) {
  $.ajax({
    url: '/predict',
    method: 'POST',
    data: {
      text: $('#src-text').val(),
      tgt: $('#tgt-dropdown').dropdown('get value')
    },
    success: function success(data) {
      $('#target-text').text(JSON.parse(data).data.output[0][0]);
      $('#more-info').removeClass('hide');
      $('#processed-text').text('Processed Text: ' + JSON.parse(data).data.output[1]);
      $('#vocab-check-en').text('English Vocab Check: ' + JSON.stringify(JSON.parse(data).data.output[2]['striped_token_list'], null, 10)); // $('#vocab-check-es').text('Spanish Vocab Check: ' + JSON.stringify(JSON.parse(data).data.output[3]['striped_token_list'], null, 10))

      $('#alignment-text').text('Attention Matrix: ');
      $('#alignment-fig').attr('src', "data:image/png;base64," + JSON.parse(data).data.output[3]);
    }
  });
});