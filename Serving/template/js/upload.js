"use strict";

/* eslint-env jquery, browser */
$('#upload-button').click(function () {
  $('#uploader').click();
});
$('#uploader').on('change', function () {
  var formData = new FormData();
  var files = document.getElementById('uploader').files;

  for (var i = 0; i < files.length; i++) {
    formData.append('file', files[i]);
  }

  $.ajax({
    url: '/upload',
    type: 'POST',
    xhr: function xhr() {
      return $.ajaxSettings.xhr();
    },
    data: formData,
    cache: false,
    contentType: false,
    processData: false,
    success: function success(data) {
      $('#input-pair').css('display', 'flex');
      $('#upload-block').css('display', 'none');
      $('#src-text').val(data.content.raw);
      $('#target-text').val(data.content.target);
      $('#download-btn').removeClass('hide');
    }
  });
});
$('#download-btn').click(function () {
  var t = $('#target-text').val();
  var encodedURI = encodeURI('data:text/plain;charset=utf-8,' + t);
  var link = document.createElement("a");
  link.setAttribute("href", encodedURI);
  link.setAttribute("download", "translation.txt");
  document.body.appendChild(link); // Required for FF

  link.click();
});